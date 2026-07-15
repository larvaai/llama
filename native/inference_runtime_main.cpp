#include "generation_safety.h"
#include "llama.h"
#include "reasoning_phase_controller.h"
#include "sampling.h"
#include "sequence_engine.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using model_worker::AdmissionStatus;
using model_worker::ControllerAction;
using model_worker::GrammarSampleSource;
using model_worker::ReasoningPhase;
using model_worker::ReasoningPhaseController;
using model_worker::SequenceEngine;
using model_worker::SequenceFinishReason;
using model_worker::SequenceLifecycle;
using model_worker::SequenceOperationStatus;
using model_worker::SequenceReleaseStatus;
using model_worker::Utf8Accumulator;

namespace {

constexpr const char * IPC_VERSION = "inference-runtime-ipc.v1";
constexpr const char * RUNTIME_MANIFEST_VERSION = "inference-runtime.v1";
constexpr std::size_t MIN_PREFIX_CACHE_TOKENS = 16;
constexpr std::size_t PREFIX_CACHE_CHECKPOINT_TOKENS = 64;

struct ModelDeleter {
    void operator()(llama_model * pointer) const { if (pointer) llama_model_free(pointer); }
};
struct ContextDeleter {
    void operator()(llama_context * pointer) const { if (pointer) llama_free(pointer); }
};
struct SamplerDeleter {
    void operator()(llama_sampler * pointer) const { if (pointer) llama_sampler_free(pointer); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

struct Batch {
    explicit Batch(const std::int32_t capacity) : value(llama_batch_init(capacity, 0, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
    llama_batch value;
};

struct RuntimeConfig {
    std::string backend_id;
    std::string model_manifest_digest;
    std::filesystem::path model_manifest_path;
    std::size_t max_sequences;
    std::size_t cpu_threads;
    std::size_t kv_tokens;
    std::size_t prefill_chunk_tokens;
    std::size_t max_decode_batch;
    std::size_t decode_quantum_tokens;
    std::size_t tick_token_budget;
    bool cache_enabled;
    std::size_t cache_byte_budget;
    std::size_t cache_max_entries;
    std::size_t cache_ttl_seconds;
};

struct NativeSessionControl {
    std::string session_id;
    std::optional<std::uint64_t> parent_generation;
    bool commit;
};

struct NativeSequenceState {
    model_worker::SequenceHandle internal;
    std::vector<llama_token> prompt;
    std::size_t logical_prefill_processed{0};
    std::size_t physical_prompt_decoded{0};
    llama_token pending_token{LLAMA_TOKEN_NULL};
    llama_pos n_past{0};
    std::unique_ptr<ReasoningPhaseController> phase;
    SamplerPtr reasoning_sampler;
    SamplerPtr final_sampler;
    std::string final_grammar_engine{"unknown"};
    bool backend_reasoning_sampler{false};
    std::string cache_identity;
    bool cache_hit{false};
    std::string cache_match{"none"};
    std::size_t cached_prompt_tokens{0};
    std::optional<NativeSessionControl> session;
    std::vector<llama_token> session_prefix;
    std::string session_commit_key;
    std::optional<std::uint64_t> committed_session_generation;
    bool session_copy_on_write{false};
    bool final_phase{false};
    Utf8Accumulator utf8;
    std::string final_bytes;
    double prompt_decode_ms{0};
    double generation_ms{0};
    std::chrono::steady_clock::time_point opened_at{
        std::chrono::steady_clock::now()
    };
    std::optional<std::chrono::steady_clock::time_point> last_sample_at;
    std::optional<std::chrono::steady_clock::time_point> last_final_at;
    double first_sample_ms{0};
    double first_final_ms{0};
    std::vector<double> sample_itl_ms;
    std::vector<double> final_itl_ms;
    std::vector<llama_token_data> sampling_candidates;
    bool terminal{false};
    bool failed{false};
    std::string error_code;
};

json read_json_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot_open_manifest");
    return json::parse(stream, nullptr, true, true);
}

void require_exact(const json & object, std::initializer_list<const char *> fields) {
    if (!object.is_object() || object.size() != fields.size()) {
        throw std::runtime_error("invalid_command_shape");
    }
    for (const auto * field : fields) {
        if (!object.contains(field)) throw std::runtime_error("invalid_command_shape");
    }
}

std::size_t positive_size(const json & value, const char * error) {
    if (!value.is_number_integer()) throw std::runtime_error(error);
    const auto converted = value.get<std::int64_t>();
    if (converted <= 0) throw std::runtime_error(error);
    return static_cast<std::size_t>(converted);
}

RuntimeConfig load_runtime_config(const std::filesystem::path & path) {
    const auto raw = read_json_file(path);
    require_exact(raw, {
        "runtime_manifest_version",
        "backend_id",
        "model_manifest",
        "model_manifest_digest",
        "scheduler",
        "cache",
    });
    if (raw.at("runtime_manifest_version") != RUNTIME_MANIFEST_VERSION) {
        throw std::runtime_error("runtime_manifest_version_mismatch");
    }
    const auto backend_id = raw.at("backend_id").get<std::string>();
    if (backend_id.empty() || backend_id.size() > 128) {
        throw std::runtime_error("backend_id_invalid");
    }
    auto model_path = std::filesystem::path(raw.at("model_manifest").get<std::string>());
    if (model_path.is_relative()) model_path = path.parent_path() / model_path;
    const auto & scheduler = raw.at("scheduler");
    require_exact(scheduler, {
        "max_sequences",
        "cpu_threads",
        "kv_tokens",
        "prefill_chunk_tokens",
        "max_decode_batch",
        "decode_quantum_tokens",
        "tick_token_budget",
    });
    const auto & cache = raw.at("cache");
    require_exact(cache, {"enabled", "byte_budget", "max_entries", "ttl_seconds"});
    if (!cache.at("enabled").is_boolean()) {
        throw std::runtime_error("cache_enabled_invalid");
    }
    RuntimeConfig result{
        backend_id,
        raw.at("model_manifest_digest").get<std::string>(),
        std::filesystem::weakly_canonical(model_path),
        positive_size(scheduler.at("max_sequences"), "max_sequences_invalid"),
        positive_size(scheduler.at("cpu_threads"), "cpu_threads_invalid"),
        positive_size(scheduler.at("kv_tokens"), "kv_tokens_invalid"),
        positive_size(scheduler.at("prefill_chunk_tokens"), "prefill_chunk_invalid"),
        positive_size(scheduler.at("max_decode_batch"), "decode_batch_invalid"),
        positive_size(scheduler.at("decode_quantum_tokens"), "decode_quantum_invalid"),
        positive_size(scheduler.at("tick_token_budget"), "tick_budget_invalid"),
        cache.at("enabled").get<bool>(),
        positive_size(cache.at("byte_budget"), "cache_byte_budget_invalid"),
        positive_size(cache.at("max_entries"), "cache_max_entries_invalid"),
        positive_size(cache.at("ttl_seconds"), "cache_ttl_invalid"),
    };
    if (
        result.max_sequences > 256
        || result.cpu_threads > 256
        || result.max_decode_batch > result.max_sequences
        || result.kv_tokens < result.max_sequences * 2
        || result.prefill_chunk_tokens > result.tick_token_budget
        || result.decode_quantum_tokens > result.tick_token_budget
        || result.max_decode_batch > result.tick_token_budget / result.decode_quantum_tokens
        || result.cache_byte_budget > 64ULL * 1024ULL * 1024ULL * 1024ULL
        || result.cache_max_entries > 65536
        || result.cache_ttl_seconds > 31536000
    ) {
        throw std::runtime_error("scheduler_config_invalid");
    }
    return result;
}

std::string bounded_cache_field(const json & value, const char * name) {
    if (!value.is_string()) throw std::runtime_error(name);
    const auto text = value.get<std::string>();
    if (text.empty() || text.size() > 256) throw std::runtime_error(name);
    for (const unsigned char character : text) {
        if (character < 0x20 || character == 0x7f) throw std::runtime_error(name);
    }
    return text;
}

std::string cache_digest_field(const json & value, const char * name) {
    const auto digest = bounded_cache_field(value, name);
    if (digest.size() != 71 || !digest.starts_with("sha256:")) {
        throw std::runtime_error(name);
    }
    for (std::size_t index = 7; index < digest.size(); ++index) {
        const auto character = digest[index];
        if (!(
            (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f')
        )) {
            throw std::runtime_error(name);
        }
    }
    return digest;
}

std::optional<std::string> cache_scope_identity(const json & value) {
    if (value.is_null()) return std::nullopt;
    require_exact(value, {"tenant_id", "workflow_id", "agent_id", "visibility"});
    const auto tenant = bounded_cache_field(value.at("tenant_id"), "cache_tenant_invalid");
    const auto workflow = bounded_cache_field(
        value.at("workflow_id"),
        "cache_workflow_invalid"
    );
    const auto agent = bounded_cache_field(value.at("agent_id"), "cache_agent_invalid");
    const auto visibility = bounded_cache_field(
        value.at("visibility"),
        "cache_visibility_invalid"
    );
    if (visibility == "tenant") return "tenant\x1f" + tenant;
    if (visibility == "workflow") {
        return "workflow\x1f" + tenant + "\x1f" + workflow;
    }
    if (visibility == "private") {
        return "private\x1f" + tenant + "\x1f" + workflow + "\x1f" + agent;
    }
    throw std::runtime_error("cache_visibility_invalid");
}

std::optional<NativeSessionControl> session_cache_control(
    const json & value,
    const json & cache_scope,
    const bool cache_enabled
) {
    if (value.is_null()) return std::nullopt;
    if (!cache_enabled || cache_scope.is_null()) {
        throw std::runtime_error("session_cache_unavailable");
    }
    require_exact(value, {"session_id", "parent_generation", "commit"});
    if (
        !cache_scope.is_object()
        || cache_scope.at("visibility") != "private"
    ) {
        throw std::runtime_error("session_cache_scope_invalid");
    }
    NativeSessionControl result{
        bounded_cache_field(value.at("session_id"), "session_id_invalid"),
        std::nullopt,
        false,
    };
    if (!value.at("parent_generation").is_null()) {
        if (!value.at("parent_generation").is_number_integer()) {
            throw std::runtime_error("session_generation_invalid");
        }
        const auto generation = value.at("parent_generation").get<std::int64_t>();
        if (generation <= 0) {
            throw std::runtime_error("session_generation_invalid");
        }
        result.parent_generation = static_cast<std::uint64_t>(generation);
    }
    if (!value.at("commit").is_boolean()) {
        throw std::runtime_error("session_commit_invalid");
    }
    result.commit = value.at("commit").get<bool>();
    return result;
}

enum class PromptCacheKind {
    prefix,
    session,
};

struct PromptCacheEntry {
    PromptCacheKind kind;
    std::string identity;
    std::string session_id;
    std::uint64_t generation;
    std::optional<std::uint64_t> parent_generation;
    std::string commit_key;
    std::vector<llama_token> tokens;
    std::vector<std::uint8_t> state;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_access;
};

struct PromptCacheRestore {
    std::size_t cached_tokens;
    bool exact;
};

struct SessionCacheStore {
    std::uint64_t generation;
    bool copy_on_write;
};

class PromptStateCache {
public:
    explicit PromptStateCache(const RuntimeConfig & config)
        : enabled_(config.cache_enabled),
          byte_budget_(config.cache_byte_budget),
          max_entries_(config.cache_max_entries),
          ttl_(std::chrono::seconds(config.cache_ttl_seconds)) {}

    std::optional<PromptCacheRestore> restore(
        const std::string & identity,
        const std::vector<llama_token> & tokens,
        llama_context * context,
        const llama_seq_id destination
    ) {
        if (!enabled_ || tokens.size() <= 1) return std::nullopt;
        const auto now = std::chrono::steady_clock::now();
        prune(now);
        PromptCacheEntry * best = nullptr;
        std::size_t best_prefix = 0;
        const auto maximum_prefix = tokens.size() - 1;
        for (auto & entry : entries_) {
            // Session snapshots are immutable lineage nodes and must only be
            // restored through restore_session() with an explicit generation.
            if (
                entry.kind != PromptCacheKind::prefix
                || entry.identity != identity
            ) {
                continue;
            }
            const auto prefix = entry.tokens.size();
            if (
                prefix < MIN_PREFIX_CACHE_TOKENS
                || prefix > maximum_prefix
                || !std::equal(
                    entry.tokens.begin(),
                    entry.tokens.end(),
                    tokens.begin()
                )
                || prefix < best_prefix
                || (
                    prefix == best_prefix
                    && best != nullptr
                    && entry.last_access <= best->last_access
                )
            ) {
                continue;
            }
            best = &entry;
            best_prefix = prefix;
        }
        if (best == nullptr) {
            ++misses_;
            return std::nullopt;
        }
        const auto restored = llama_state_seq_set_data(
            context,
            best->state.data(),
            best->state.size(),
            destination
        );
        if (restored == 0) {
            ++restore_failures_;
            ++misses_;
            return std::nullopt;
        }
        best->last_access = now;
        ++hits_;
        const bool exact = best_prefix == maximum_prefix;
        if (exact) {
            ++exact_hits_;
        } else {
            ++prefix_hits_;
        }
        saved_prefill_tokens_ += best_prefix;
        return PromptCacheRestore{best_prefix, exact};
    }

    bool store(
        const std::string & identity,
        const std::vector<llama_token> & tokens,
        llama_context * context,
        const llama_seq_id source
    ) {
        if (!enabled_ || tokens.empty()) return false;
        const auto now = std::chrono::steady_clock::now();
        prune(now);
        for (auto & entry : entries_) {
            if (
                entry.kind == PromptCacheKind::prefix
                && entry.identity == identity
                && entry.tokens == tokens
            ) {
                entry.last_access = now;
                return true;
            }
        }
        PromptCacheEntry entry{
            PromptCacheKind::prefix,
            identity,
            {},
            0,
            std::nullopt,
            {},
            tokens,
            {},
            now,
            now,
        };
        if (!capture(std::move(entry), context, source)) return false;
        ++insertions_;
        return true;
    }

    std::optional<PromptCacheRestore> restore_session(
        const std::string & identity,
        const std::string & session_id,
        const std::uint64_t generation,
        const std::vector<llama_token> & tokens,
        llama_context * context,
        const llama_seq_id destination
    ) {
        if (!enabled_ || tokens.size() <= 1) return std::nullopt;
        const auto now = std::chrono::steady_clock::now();
        prune(now);
        auto entry = std::find_if(
            entries_.begin(),
            entries_.end(),
            [&](const auto & candidate) {
                return candidate.kind == PromptCacheKind::session
                    && candidate.identity == identity
                    && candidate.session_id == session_id
                    && candidate.generation == generation;
            }
        );
        if (entry == entries_.end() || entry->tokens.size() > tokens.size() - 1) {
            ++session_misses_;
            return std::nullopt;
        }
        if (!std::equal(entry->tokens.begin(), entry->tokens.end(), tokens.begin())) {
            ++session_misses_;
            return std::nullopt;
        }
        const auto restored = llama_state_seq_set_data(
            context,
            entry->state.data(),
            entry->state.size(),
            destination
        );
        if (restored == 0) {
            ++restore_failures_;
            ++session_misses_;
            return std::nullopt;
        }
        entry->last_access = now;
        ++hits_;
        ++session_hits_;
        saved_prefill_tokens_ += entry->tokens.size();
        return PromptCacheRestore{
            entry->tokens.size(),
            entry->tokens.size() == tokens.size() - 1,
        };
    }

    bool session_root_allowed(
        const std::string & identity,
        const std::string & session_id,
        const std::string & commit_key
    ) {
        prune(std::chrono::steady_clock::now());
        return std::none_of(
            entries_.begin(),
            entries_.end(),
            [&](const auto & entry) {
                return entry.kind == PromptCacheKind::session
                    && entry.identity == identity
                    && entry.session_id == session_id
                    && entry.commit_key != commit_key;
            }
        );
    }

    std::optional<SessionCacheStore> store_session(
        const std::string & identity,
        const std::string & session_id,
        const std::optional<std::uint64_t> parent_generation,
        const std::string & commit_key,
        const std::vector<llama_token> & tokens,
        llama_context * context,
        const llama_seq_id source
    ) {
        if (!enabled_ || tokens.empty()) return std::nullopt;
        const auto now = std::chrono::steady_clock::now();
        prune(now);
        for (auto & entry : entries_) {
            if (
                entry.kind == PromptCacheKind::session
                && entry.identity == identity
                && entry.session_id == session_id
                && entry.commit_key == commit_key
            ) {
                if (
                    entry.parent_generation != parent_generation
                    || entry.tokens != tokens
                ) {
                    ++store_failures_;
                    return std::nullopt;
                }
                entry.last_access = now;
                return SessionCacheStore{entry.generation, false};
            }
        }
        std::uint64_t generation = 1;
        bool copy_on_write = false;
        for (const auto & entry : entries_) {
            if (
                entry.kind != PromptCacheKind::session
                || entry.identity != identity
                || entry.session_id != session_id
            ) {
                continue;
            }
            if (!parent_generation.has_value()) {
                ++store_failures_;
                return std::nullopt;
            }
            generation = std::max(generation, entry.generation + 1);
            if (
                parent_generation.has_value()
                && entry.parent_generation == parent_generation
            ) {
                copy_on_write = true;
            }
        }
        PromptCacheEntry entry{
            PromptCacheKind::session,
            identity,
            session_id,
            generation,
            parent_generation,
            commit_key,
            tokens,
            {},
            now,
            now,
        };
        if (!capture(std::move(entry), context, source)) return std::nullopt;
        ++insertions_;
        ++session_insertions_;
        if (copy_on_write) ++cow_clones_;
        return SessionCacheStore{generation, copy_on_write};
    }

    json stats() {
        prune(std::chrono::steady_clock::now());
        return {
            {"enabled", enabled_},
            {"entries", entries_.size()},
            {"bytes_used", bytes_used_},
            {"byte_budget", byte_budget_},
            {"hits", hits_},
            {"exact_hits", exact_hits_},
            {"prefix_hits", prefix_hits_},
            {"session_hits", session_hits_},
            {"misses", misses_},
            {"session_misses", session_misses_},
            {"insertions", insertions_},
            {"session_insertions", session_insertions_},
            {"session_entries", std::count_if(
                entries_.begin(),
                entries_.end(),
                [](const auto & entry) {
                    return entry.kind == PromptCacheKind::session;
                }
            )},
            {"cow_clones", cow_clones_},
            {"evictions", evictions_},
            {"restore_failures", restore_failures_},
            {"store_failures", store_failures_},
            {"saved_prefill_tokens", saved_prefill_tokens_},
        };
    }

    std::size_t clear() {
        const auto removed = entries_.size();
        entries_.clear();
        bytes_used_ = 0;
        return removed;
    }

private:
    bool capture(
        PromptCacheEntry entry,
        llama_context * context,
        const llama_seq_id source
    ) {
        const auto size = llama_state_seq_get_size(context, source);
        if (size == 0 || size > byte_budget_) {
            ++store_failures_;
            return false;
        }
        while (
            !entries_.empty()
            && (bytes_used_ + size > byte_budget_ || entries_.size() >= max_entries_)
        ) {
            const auto victim = std::min_element(
                entries_.begin(),
                entries_.end(),
                [](const auto & left, const auto & right) {
                    return left.last_access < right.last_access;
                }
            );
            bytes_used_ -= victim->state.size();
            entries_.erase(victim);
            ++evictions_;
        }
        if (bytes_used_ + size > byte_budget_ || entries_.size() >= max_entries_) {
            ++store_failures_;
            return false;
        }
        entry.state.resize(size);
        const auto copied = llama_state_seq_get_data(
            context,
            entry.state.data(),
            entry.state.size(),
            source
        );
        if (copied == 0 || copied > entry.state.size()) {
            ++store_failures_;
            return false;
        }
        entry.state.resize(copied);
        bytes_used_ += entry.state.size();
        entries_.push_back(std::move(entry));
        return true;
    }

    void prune(const std::chrono::steady_clock::time_point now) {
        auto iterator = entries_.begin();
        while (iterator != entries_.end()) {
            if (now - iterator->last_access < ttl_) {
                ++iterator;
                continue;
            }
            bytes_used_ -= iterator->state.size();
            iterator = entries_.erase(iterator);
            ++evictions_;
        }
    }

    bool enabled_;
    std::size_t byte_budget_;
    std::size_t max_entries_;
    std::chrono::seconds ttl_;
    std::vector<PromptCacheEntry> entries_;
    std::size_t bytes_used_{0};
    std::uint64_t hits_{0};
    std::uint64_t exact_hits_{0};
    std::uint64_t prefix_hits_{0};
    std::uint64_t session_hits_{0};
    std::uint64_t misses_{0};
    std::uint64_t session_misses_{0};
    std::uint64_t insertions_{0};
    std::uint64_t session_insertions_{0};
    std::uint64_t cow_clones_{0};
    std::uint64_t evictions_{0};
    std::uint64_t restore_failures_{0};
    std::uint64_t store_failures_{0};
    std::uint64_t saved_prefill_tokens_{0};
};

std::vector<llama_token> tokenize(
    const llama_vocab * vocab,
    const std::string & text,
    const bool add_special,
    const bool parse_special
) {
    int32_t count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        nullptr,
        0,
        add_special,
        parse_special
    );
    if (count == 0) return {};
    if (count > 0) throw std::runtime_error("tokenizer_size_query_failed");
    std::vector<llama_token> result(static_cast<std::size_t>(-count));
    count = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        result.data(),
        static_cast<int32_t>(result.size()),
        add_special,
        parse_special
    );
    if (count < 0) throw std::runtime_error("tokenization_failed");
    result.resize(static_cast<std::size_t>(count));
    return result;
}

std::string token_piece(const llama_vocab * vocab, const llama_token token) {
    std::vector<char> buffer(32);
    int32_t count = llama_token_to_piece(
        vocab,
        token,
        buffer.data(),
        static_cast<int32_t>(buffer.size()),
        0,
        true
    );
    if (count < 0) {
        buffer.resize(static_cast<std::size_t>(-count));
        count = llama_token_to_piece(
            vocab,
            token,
            buffer.data(),
            static_cast<int32_t>(buffer.size()),
            0,
            true
        );
    }
    return count > 0
        ? std::string(buffer.data(), static_cast<std::size_t>(count))
        : std::string{};
}

std::vector<llama_token> build_prompt(
    llama_model * model,
    const llama_vocab * vocab,
    const json & model_messages,
    const bool add_generation_prompt = true
) {
    if (!model_messages.is_array() || model_messages.empty()) {
        throw std::runtime_error("model_messages_invalid");
    }
    std::vector<std::string> roles;
    std::vector<std::string> contents;
    std::vector<llama_chat_message> messages;
    roles.reserve(model_messages.size());
    contents.reserve(model_messages.size());
    for (const auto & item : model_messages) {
        require_exact(item, {"role", "content"});
        roles.push_back(item.at("role").get<std::string>());
        contents.push_back(item.at("content").get<std::string>());
    }
    messages.reserve(roles.size());
    for (std::size_t index = 0; index < roles.size(); ++index) {
        messages.push_back({roles[index].c_str(), contents[index].c_str()});
    }
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) throw std::runtime_error("chat_template_missing");
    const int32_t needed = llama_chat_apply_template(
        chat_template,
        messages.data(),
        messages.size(),
        add_generation_prompt,
        nullptr,
        0
    );
    if (needed <= 0) throw std::runtime_error("chat_template_failed");
    std::vector<char> buffer(static_cast<std::size_t>(needed) + 1);
    const int32_t written = llama_chat_apply_template(
        chat_template,
        messages.data(),
        messages.size(),
        add_generation_prompt,
        buffer.data(),
        static_cast<int32_t>(buffer.size())
    );
    if (written != needed) throw std::runtime_error("chat_template_failed");
    return tokenize(vocab, std::string(buffer.data(), written), true, true);
}

void batch_add(
    llama_batch & batch,
    const llama_token token,
    const llama_pos position,
    const llama_seq_id sequence,
    const bool logits
) {
    const auto index = batch.n_tokens;
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence;
    batch.logits[index] = logits ? 1 : 0;
    ++batch.n_tokens;
}

json public_handle(
    const RuntimeConfig & config,
    const std::string & model_id,
    const model_worker::SequenceHandle & handle
) {
    return {
        {"backend", config.backend_id},
        {"model", model_id},
        {"sequence", "slot-" + std::to_string(handle.slot)},
        {"generation", handle.generation},
    };
}

std::optional<std::pair<std::size_t, std::uint64_t>> parse_public_handle(
    const json & raw,
    const RuntimeConfig & config,
    const std::string & model_id
) {
    try {
        require_exact(raw, {"backend", "model", "sequence", "generation"});
        if (raw.at("backend") != config.backend_id || raw.at("model") != model_id) {
            return std::nullopt;
        }
        const auto sequence = raw.at("sequence").get<std::string>();
        if (!sequence.starts_with("slot-") || sequence.size() <= 5) return std::nullopt;
        std::size_t consumed = 0;
        const auto slot = std::stoull(sequence.substr(5), &consumed);
        if (consumed != sequence.size() - 5 || slot >= config.max_sequences) {
            return std::nullopt;
        }
        if (!raw.at("generation").is_number_integer()) return std::nullopt;
        const auto generation = raw.at("generation").get<std::uint64_t>();
        if (generation == 0) return std::nullopt;
        return std::make_pair(static_cast<std::size_t>(slot), generation);
    } catch (...) {
        return std::nullopt;
    }
}

void emit(const json & frame) {
    std::cout << frame.dump(-1, ' ', false, json::error_handler_t::replace)
              << '\n' << std::flush;
}

json response_base(const std::string & type, const std::uint64_t command_id) {
    return {
        {"protocol_version", IPC_VERSION},
        {"type", type},
        {"command_id", command_id},
    };
}

void emit_error(
    const std::uint64_t command_id,
    const std::string & code,
    const std::string & detail
) {
    auto frame = response_base("command_error", command_id);
    frame["error_code"] = code;
    frame["detail"] = detail;
    emit(frame);
}

std::uint64_t command_id_of(const json & command) {
    if (!command.contains("command_id") || !command.at("command_id").is_number_integer()) {
        throw std::runtime_error("command_id_invalid");
    }
    return command.at("command_id").get<std::uint64_t>();
}

SamplerPtr greedy_sampler() {
    auto parameters = llama_sampler_chain_default_params();
    parameters.no_perf = true;
    SamplerPtr sampler(llama_sampler_chain_init(parameters));
    if (!sampler) throw std::runtime_error("sampler_init_failed");
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    return sampler;
}

SamplerPtr grammar_sampler(
    const llama_vocab * vocab,
    const json & schema,
    const std::string & grammar_text,
    std::string & engine
) {
    auto parameters = llama_sampler_chain_default_params();
    parameters.no_perf = true;
    SamplerPtr sampler(llama_sampler_chain_init(parameters));
    if (!sampler) throw std::runtime_error("sampler_init_failed");
    const auto llguidance_grammar = std::string("%llguidance {}\nstart: %json ")
        + schema.dump();
    auto * grammar = llama_sampler_init_llg(
        vocab,
        "lark",
        llguidance_grammar.c_str()
    );
    if (!grammar) {
        grammar = llama_sampler_init_grammar(vocab, grammar_text.c_str(), "root");
    }
    if (!grammar) throw std::runtime_error("grammar_init_failed");
    engine = llama_sampler_name(grammar);
    llama_sampler_chain_add(sampler.get(), grammar);
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    return sampler;
}

llama_token sample_from_logits(
    llama_sampler * sampler,
    const float * logits,
    const std::int32_t n_vocab,
    std::vector<llama_token_data> & candidates,
    const bool grammar_rejection
) {
    if (!sampler || !logits || n_vocab <= 0) {
        throw std::runtime_error("sampling_input_invalid");
    }
    llama_token token = 0;
    for (llama_token candidate = 1; candidate < n_vocab; ++candidate) {
        if (logits[candidate] > logits[token]) token = candidate;
    }
    if (grammar_rejection) {
        auto * grammar = llama_sampler_chain_get(sampler, 0);
        if (!grammar) throw std::runtime_error("grammar_sampler_missing");
        llama_token_data candidate{token, logits[token], 0.0F};
        llama_token_data_array single{
            &candidate,
            1,
            -1,
            false,
        };
        llama_sampler_apply(grammar, &single);
        if (candidate.logit != -std::numeric_limits<float>::infinity()) {
            llama_sampler_accept(sampler, token);
            return token;
        }
    } else {
        llama_sampler_accept(sampler, token);
        return token;
    }

    // The unconstrained greedy token was rejected. Preserve exact
    // grammar-first greedy semantics by masking the full candidate set only
    // on this slow path, matching llama.cpp common_sampler rejection sampling.
    candidates.resize(static_cast<std::size_t>(n_vocab));
    for (llama_token candidate = 0; candidate < n_vocab; ++candidate) {
        candidates[static_cast<std::size_t>(candidate)] = {
            candidate,
            logits[candidate],
            0.0F,
        };
    }
    llama_token_data_array view{
        candidates.data(),
        candidates.size(),
        -1,
        false,
    };
    llama_sampler_apply(sampler, &view);
    if (view.selected < 0 || view.selected >= static_cast<std::int64_t>(view.size)) {
        throw std::runtime_error("sampling_selection_invalid");
    }
    token = view.data[view.selected].id;
    llama_sampler_accept(sampler, token);
    return token;
}

std::string admission_code(const AdmissionStatus status) {
    switch (status) {
        case AdmissionStatus::duplicate: return "duplicate_sequence";
        case AdmissionStatus::sequence_capacity: return "sequence_capacity";
        case AdmissionStatus::token_capacity: return "token_capacity";
        default: return "invalid_sequence_request";
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: inference-runtime-native <runtime-manifest>\n";
        return 64;
    }
    try {
        const auto runtime_path = std::filesystem::weakly_canonical(argv[1]);
        const auto config = load_runtime_config(runtime_path);
        const auto model_manifest = read_json_file(config.model_manifest_path);
        if (
            model_manifest.at("manifest_version") != "model-manifest.v1"
            || model_manifest.at("runtime_build") != "b10012"
        ) {
            throw std::runtime_error("model_manifest_version_mismatch");
        }
        const auto model_id = model_manifest.at("id").get<std::string>();
        const auto & reasoning = model_manifest.at("reasoning");
        if (
            reasoning.value("mode", "") != "required_marker_sequence"
            || reasoning.value("require_start", false) != true
        ) {
            throw std::runtime_error("reasoning_capability_invalid");
        }
        const auto & context_config = model_manifest.at("context");
        const auto n_batch = positive_size(context_config.at("n_batch"), "n_batch_invalid");
        const auto n_ubatch = positive_size(context_config.at("n_ubatch"), "n_ubatch_invalid");
        const auto per_sequence_context = positive_size(
            context_config.at("n_ctx"),
            "n_ctx_invalid"
        );
        const auto sequence_context_tokens = std::min(
            per_sequence_context,
            config.kv_tokens / config.max_sequences
        );
        if (
            config.tick_token_budget > n_batch
            || config.prefill_chunk_tokens > n_batch
            || config.kv_tokens > std::numeric_limits<std::uint32_t>::max()
        ) {
            throw std::runtime_error("runtime_context_invalid");
        }

        llama_backend_init();
        auto model_parameters = llama_model_default_params();
        model_parameters.n_gpu_layers = model_manifest.at("gpu").at("layers");
        ModelPtr model(llama_model_load_from_file(
            model_manifest.at("gguf_path").get_ref<const std::string &>().c_str(),
            model_parameters
        ));
        if (!model) throw std::runtime_error("model_load_failed");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const auto vocabulary_size = llama_vocab_n_tokens(vocab);
        if (vocabulary_size <= 0) throw std::runtime_error("model_vocab_invalid");
        const auto start_marker = tokenize(
            vocab,
            reasoning.at("start_text").get_ref<const std::string &>(),
            false,
            true
        );
        const auto end_marker = tokenize(
            vocab,
            reasoning.at("end_text").get_ref<const std::string &>(),
            false,
            true
        );
        if (
            start_marker.empty()
            || end_marker.empty()
            || start_marker == end_marker
            || !llama_model_chat_template(model.get(), nullptr)
            || llama_model_n_ctx_train(model.get()) < static_cast<int64_t>(per_sequence_context)
        ) {
            throw std::runtime_error("capability_verification_failed");
        }

        auto context_parameters = llama_context_default_params();
        context_parameters.n_ctx = static_cast<std::uint32_t>(config.kv_tokens);
        context_parameters.n_batch = static_cast<std::uint32_t>(n_batch);
        context_parameters.n_ubatch = static_cast<std::uint32_t>(n_ubatch);
        context_parameters.n_seq_max = static_cast<std::uint32_t>(config.max_sequences);
        context_parameters.n_threads = static_cast<std::int32_t>(config.cpu_threads);
        context_parameters.n_threads_batch = static_cast<std::int32_t>(config.cpu_threads);
        // At most one logit row per sequence is requested on a decode tick.
        // Leaving this at zero reserves n_batch rows (1024 here) and makes the
        // output graph materially slower than llama-server's slot runtime.
        context_parameters.n_outputs_max = static_cast<std::uint32_t>(
            config.max_decode_batch
        );
        // Match llama-server's production default. A full-size SWA cache adds
        // avoidable attention work for independent short agent sequences.
        context_parameters.swa_full = false;
        context_parameters.no_perf = true;
        ContextPtr context(llama_init_from_model(model.get(), context_parameters));
        if (!context) throw std::runtime_error("context_init_failed");

        SequenceEngine engine(config.max_sequences, config.kv_tokens);
        std::vector<std::unique_ptr<NativeSequenceState>> states(config.max_sequences);
        PromptStateCache prompt_cache(config);
        emit({
            {"protocol_version", IPC_VERSION},
            {"type", "ready"},
            {"sequence", 0},
            {"backend_id", config.backend_id},
            {"model_id", model_id},
            {"model_manifest_digest", config.model_manifest_digest},
            {"max_sequences", config.max_sequences},
            {"cpu_threads", config.cpu_threads},
            {"kv_tokens", config.kv_tokens},
            {"sequence_context_tokens", sequence_context_tokens},
            {"prefill_chunk_tokens", config.prefill_chunk_tokens},
            {"max_decode_batch", config.max_decode_batch},
            {"decode_quantum_tokens", config.decode_quantum_tokens},
            {"tick_token_budget", config.tick_token_budget},
            {"cache", {
                {"enabled", config.cache_enabled},
                {"byte_budget", config.cache_byte_budget},
                {"max_entries", config.cache_max_entries},
                {"ttl_seconds", config.cache_ttl_seconds},
            }},
        });

        std::string line;
        while (std::getline(std::cin, line)) {
            std::uint64_t command_id = 0;
            try {
                const auto command = json::parse(line, nullptr, true, true);
                command_id = command_id_of(command);
                if (command.value("protocol_version", "") != IPC_VERSION) {
                    throw std::runtime_error("protocol_version_mismatch");
                }
                const auto type = command.value("type", "");

                if (type == "shutdown") {
                    require_exact(command, {"protocol_version", "type", "command_id"});
                    auto response = response_base("shutdown_complete", command_id);
                    emit(response);
                    break;
                }

                if (type == "cache_stats") {
                    require_exact(command, {"protocol_version", "type", "command_id"});
                    auto response = response_base("cache_stats", command_id);
                    response["cache"] = prompt_cache.stats();
                    emit(response);
                    continue;
                }

                if (type == "cache_clear") {
                    require_exact(command, {"protocol_version", "type", "command_id"});
                    auto response = response_base("cache_cleared", command_id);
                    response["removed_entries"] = prompt_cache.clear();
                    emit(response);
                    continue;
                }

                if (type == "open_sequence") {
                    require_exact(command, {
                        "protocol_version",
                        "type",
                        "command_id",
                        "request_id",
                        "attempt_id",
                        "request",
                    });
                    const auto request_id = command.at("request_id").get<std::string>();
                    const auto attempt_id = command.at("attempt_id").get<std::string>();
                    if (request_id.empty() || attempt_id.empty()) {
                        throw std::runtime_error("request_identity_invalid");
                    }
                    const auto & envelope = command.at("request");
                    if (
                        !envelope.is_object()
                        || !envelope.contains("request")
                        || !envelope.contains("grammar")
                        || !envelope.contains("model_messages")
                        || !envelope.contains("prompt_hash")
                        || !envelope.contains("prompt_version")
                        || !envelope.contains("cache_scope")
                        || !envelope.contains("session_cache")
                        || !envelope.contains("cache_namespace")
                    ) {
                        throw std::runtime_error("request_envelope_invalid");
                    }
                    const auto & request = envelope.at("request");
                    const auto & limits = request.at("limits");
                    const auto total_tokens = positive_size(
                        limits.at("total_tokens"),
                        "total_tokens_invalid"
                    );
                    const auto reasoning_tokens = positive_size(
                        limits.at("reasoning_tokens"),
                        "reasoning_tokens_invalid"
                    );
                    const auto final_tokens = positive_size(
                        limits.at("final_tokens"),
                        "final_tokens_invalid"
                    );
                    auto prompt = build_prompt(
                        model.get(),
                        vocab,
                        envelope.at("model_messages")
                    );
                    const auto & cache_scope_value = envelope.at("cache_scope");
                    const auto cache_scope = cache_scope_identity(cache_scope_value);
                    const auto session = session_cache_control(
                        envelope.at("session_cache"),
                        cache_scope_value,
                        config.cache_enabled
                    );
                    std::vector<llama_token> session_prefix;
                    if (session.has_value()) {
                        session_prefix = build_prompt(
                            model.get(),
                            vocab,
                            envelope.at("model_messages"),
                            false
                        );
                        if (
                            session_prefix.empty()
                            || session_prefix.size() > prompt.size() - 1
                            || !std::equal(
                                session_prefix.begin(),
                                session_prefix.end(),
                                prompt.begin()
                            )
                        ) {
                            throw std::runtime_error("session_prefix_invalid");
                        }
                    }
                    bounded_cache_field(
                        envelope.at("prompt_hash"),
                        "prompt_hash_invalid"
                    );
                    const auto prompt_version = bounded_cache_field(
                        envelope.at("prompt_version"),
                        "prompt_version_invalid"
                    );
                    const auto & cache_namespace = envelope.at("cache_namespace");
                    require_exact(cache_namespace, {
                        "model_digest",
                        "template_digest",
                        "tokenizer_digest",
                        "adapter_digest",
                        "context_digest",
                    });
                    const auto cache_model_digest = cache_digest_field(
                        cache_namespace.at("model_digest"),
                        "cache_model_digest_invalid"
                    );
                    if (cache_model_digest != config.model_manifest_digest) {
                        throw std::runtime_error("cache_model_digest_mismatch");
                    }
                    const auto cache_template_digest = cache_digest_field(
                        cache_namespace.at("template_digest"),
                        "cache_template_digest_invalid"
                    );
                    const auto cache_tokenizer_digest = cache_digest_field(
                        cache_namespace.at("tokenizer_digest"),
                        "cache_tokenizer_digest_invalid"
                    );
                    const auto cache_adapter_digest = cache_digest_field(
                        cache_namespace.at("adapter_digest"),
                        "cache_adapter_digest_invalid"
                    );
                    const auto cache_context_digest = cache_digest_field(
                        cache_namespace.at("context_digest"),
                        "cache_context_digest_invalid"
                    );
                    if (
                        prompt.empty()
                        || prompt.size() + total_tokens + 8 > sequence_context_tokens
                    ) {
                        emit_error(command_id, "context_overflow", "per-sequence context exceeded");
                        continue;
                    }
                    auto state = std::make_unique<NativeSequenceState>();
                    state->prompt = std::move(prompt);
                    state->session = session;
                    state->session_prefix = std::move(session_prefix);
                    state->session_commit_key = request_id;
                    state->pending_token = state->prompt.back();
                    state->phase = std::make_unique<ReasoningPhaseController>(
                        start_marker,
                        end_marker,
                        reasoning_tokens,
                        final_tokens,
                        total_tokens,
                        true
                    );
                    state->reasoning_sampler = greedy_sampler();
                    state->final_sampler = grammar_sampler(
                        vocab,
                        request.at("output_contract").at("schema"),
                        envelope.at("grammar").get_ref<const std::string &>(),
                        state->final_grammar_engine
                    );
                    if (config.cache_enabled && cache_scope.has_value()) {
                        state->cache_identity = *cache_scope
                            + "\x1e" + cache_model_digest
                            + "\x1e" + cache_template_digest
                            + "\x1e" + cache_tokenizer_digest
                            + "\x1e" + cache_adapter_digest
                            + "\x1e" + cache_context_digest
                            + "\x1e" + prompt_version
                            + "\x1e" + std::to_string(sequence_context_tokens)
                            + "\x1e" + std::to_string(config.kv_tokens)
                            + "\x1e" + std::to_string(config.max_sequences);
                    }
                    if (
                        state->session.has_value()
                        && !state->session->parent_generation.has_value()
                        && !prompt_cache.session_root_allowed(
                            state->cache_identity,
                            state->session->session_id,
                            request_id
                        )
                    ) {
                        emit_error(
                            command_id,
                            "session_conflict",
                            "session already has a root snapshot"
                        );
                        continue;
                    }
                    const auto admitted = engine.admit(
                        request_id,
                        attempt_id,
                        state->prompt.size(),
                        total_tokens
                    );
                    if (!admitted.handle.has_value()) {
                        emit_error(
                            command_id,
                            admission_code(admitted.status),
                            "sequence admission rejected"
                        );
                        continue;
                    }
                    state->internal = *admitted.handle;
                    if (engine.start_prefill(state->internal) != SequenceOperationStatus::applied) {
                        throw std::runtime_error("sequence_state_invalid");
                    }
                    const auto slot = static_cast<std::size_t>(state->internal.slot);
                    const auto handle_json = public_handle(config, model_id, state->internal);
                    const auto reservation = state->prompt.size() + total_tokens;
                    states[slot] = std::move(state);
                    // A constrained sequence must expose CPU logits in its final phase.
                    // Keeping one stable logits graph avoids the expensive graph rebuild
                    // caused by switching away from experimental backend sampling mid-run.
                    states[slot]->backend_reasoning_sampler = false;
                    if (!states[slot]->cache_identity.empty()) {
                        const auto restored = (
                            states[slot]->session.has_value()
                            && states[slot]->session->parent_generation.has_value()
                        )
                            ? prompt_cache.restore_session(
                                states[slot]->cache_identity,
                                states[slot]->session->session_id,
                                *states[slot]->session->parent_generation,
                                states[slot]->prompt,
                                context.get(),
                                states[slot]->internal.slot
                            )
                            : prompt_cache.restore(
                                states[slot]->cache_identity,
                                states[slot]->prompt,
                                context.get(),
                                states[slot]->internal.slot
                            );
                        if (
                            states[slot]->session.has_value()
                            && states[slot]->session->parent_generation.has_value()
                            && !restored.has_value()
                        ) {
                            engine.cancel(states[slot]->internal);
                            engine.release(states[slot]->internal);
                            states[slot].reset();
                            emit_error(
                                command_id,
                                "session_snapshot_missing",
                                "session parent is missing or not a prompt prefix"
                            );
                            continue;
                        }
                        if (restored.has_value()) {
                            states[slot]->cache_hit = true;
                            states[slot]->cache_match = states[slot]->session.has_value()
                                && states[slot]->session->parent_generation.has_value()
                                    ? "session"
                                    : (restored->exact ? "exact" : "prefix");
                            states[slot]->logical_prefill_processed =
                                restored->cached_tokens;
                            states[slot]->physical_prompt_decoded =
                                restored->cached_tokens;
                            states[slot]->n_past = static_cast<llama_pos>(
                                states[slot]->physical_prompt_decoded
                            );
                            states[slot]->cached_prompt_tokens =
                                states[slot]->physical_prompt_decoded;
                            if (
                                engine.advance_prefill(
                                    states[slot]->internal,
                                    restored->cached_tokens
                                ) != SequenceOperationStatus::applied
                            ) {
                                throw std::runtime_error("sequence_state_invalid");
                            }
                        }
                    }
                    auto response = response_base("sequence_opened", command_id);
                    response["handle"] = handle_json;
                    response["prompt_tokens"] = states[slot]->prompt.size();
                    response["reserved_tokens"] = reservation;
                    response["cache_hit"] = states[slot]->cache_hit;
                    response["cached_prompt_tokens"] =
                        states[slot]->cached_prompt_tokens;
                    emit(response);
                    continue;
                }

                if (type == "prefill_batch") {
                    require_exact(command, {"protocol_version", "type", "command_id", "steps"});
                    const auto & steps = command.at("steps");
                    if (
                        !steps.is_array()
                        || steps.empty()
                        || steps.size() > config.max_sequences
                    ) {
                        throw std::runtime_error("prefill_steps_invalid");
                    }
                    struct Work {
                        NativeSequenceState * state;
                        json handle;
                        std::size_t logical_tokens;
                        std::size_t physical_tokens;
                    };
                    std::vector<Work> work;
                    std::vector<bool> seen(config.max_sequences, false);
                    std::size_t batch_tokens = 0;
                    for (const auto & step : steps) {
                        require_exact(step, {"handle", "token_budget"});
                        const auto parsed = parse_public_handle(
                            step.at("handle"), config, model_id
                        );
                        if (!parsed.has_value() || parsed->first >= states.size()) {
                            throw std::runtime_error("stale_handle");
                        }
                        if (seen[parsed->first]) {
                            throw std::runtime_error("duplicate_sequence_step");
                        }
                        seen[parsed->first] = true;
                        auto * state = states[parsed->first].get();
                        if (
                            state == nullptr
                            || state->internal.generation != parsed->second
                            || engine.snapshot(state->internal)->lifecycle
                                != SequenceLifecycle::prefill
                        ) {
                            throw std::runtime_error("stale_handle");
                        }
                        const auto budget = positive_size(
                            step.at("token_budget"),
                            "prefill_budget_invalid"
                        );
                        if (budget > config.prefill_chunk_tokens) {
                            throw std::runtime_error("prefill_budget_invalid");
                        }
                        const auto remaining = state->prompt.size()
                            - state->logical_prefill_processed;
                        auto logical = std::min(budget, remaining);
                        if (!state->cache_identity.empty()) {
                            if (
                                state->session.has_value()
                                && state->session->commit
                                && !state->committed_session_generation.has_value()
                                && state->physical_prompt_decoded
                                    < state->session_prefix.size()
                            ) {
                                logical = std::min(
                                    logical,
                                    state->session_prefix.size()
                                        - state->physical_prompt_decoded
                                );
                            } else if (!state->session.has_value()) {
                                const auto checkpoint_remaining =
                                    PREFIX_CACHE_CHECKPOINT_TOKENS
                                    - (
                                        state->physical_prompt_decoded
                                        % PREFIX_CACHE_CHECKPOINT_TOKENS
                                    );
                                logical = std::min(logical, checkpoint_remaining);
                            }
                        }
                        const auto reaches_end = logical == remaining;
                        const auto physical = reaches_end && logical > 0 ? logical - 1 : logical;
                        batch_tokens += physical;
                        work.push_back({state, step.at("handle"), logical, physical});
                    }
                    if (batch_tokens > config.tick_token_budget || batch_tokens > n_batch) {
                        throw std::runtime_error("tick_token_budget_exceeded");
                    }
                    Batch batch(static_cast<std::int32_t>(std::max<std::size_t>(1, batch_tokens)));
                    for (auto & item : work) {
                        for (std::size_t offset = 0; offset < item.physical_tokens; ++offset) {
                            const auto prompt_index = item.state->physical_prompt_decoded + offset;
                            batch_add(
                                batch.value,
                                item.state->prompt[prompt_index],
                                static_cast<llama_pos>(prompt_index),
                                item.state->internal.slot,
                                false
                            );
                        }
                    }
                    const auto started = std::chrono::steady_clock::now();
                    if (batch.value.n_tokens > 0 && llama_decode(context.get(), batch.value) != 0) {
                        throw std::runtime_error("decode_failed");
                    }
                    if (batch.value.n_tokens > 0) {
                        llama_synchronize(context.get());
                    }
                    const auto elapsed = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started
                    ).count();
                    auto outcomes = json::array();
                    for (auto & item : work) {
                        item.state->physical_prompt_decoded += item.physical_tokens;
                        item.state->logical_prefill_processed += item.logical_tokens;
                        item.state->n_past = static_cast<llama_pos>(
                            item.state->physical_prompt_decoded
                        );
                        if (item.physical_tokens > 0) {
                            item.state->prompt_decode_ms += elapsed;
                        }
                        if (
                            engine.advance_prefill(
                                item.state->internal,
                                item.logical_tokens
                            )
                                != SequenceOperationStatus::applied
                        ) {
                            throw std::runtime_error("sequence_state_invalid");
                        }
                        const auto remaining = item.state->prompt.size()
                            - item.state->logical_prefill_processed;
                        if (!item.state->cache_identity.empty()) {
                            const auto physical_end =
                                item.state->physical_prompt_decoded;
                            if (
                                item.state->session.has_value()
                                && item.state->session->commit
                                && !item.state->committed_session_generation.has_value()
                                && physical_end == item.state->session_prefix.size()
                            ) {
                                const auto committed = prompt_cache.store_session(
                                    item.state->cache_identity,
                                    item.state->session->session_id,
                                    item.state->session->parent_generation,
                                    item.state->session_commit_key,
                                    item.state->session_prefix,
                                    context.get(),
                                    item.state->internal.slot
                                );
                                if (!committed.has_value()) {
                                    throw std::runtime_error("session_commit_failed");
                                }
                                item.state->committed_session_generation =
                                    committed->generation;
                                item.state->session_copy_on_write =
                                    committed->copy_on_write;
                            } else if (
                                !item.state->session.has_value()
                                && physical_end > 0
                            ) {
                                const std::vector<llama_token> cached_tokens(
                                    item.state->prompt.begin(),
                                    item.state->prompt.begin() + physical_end
                                );
                                prompt_cache.store(
                                    item.state->cache_identity,
                                    cached_tokens,
                                    context.get(),
                                    item.state->internal.slot
                                );
                            }
                        }
                        outcomes.push_back({
                            {"handle", item.handle},
                            {"status", remaining == 0 ? "ready" : "partial"},
                            {"processed_tokens", item.logical_tokens},
                            {"remaining_tokens", remaining},
                        });
                    }
                    auto response = response_base("prefill_completed", command_id);
                    response["outcomes"] = std::move(outcomes);
                    emit(response);
                    continue;
                }

                if (type == "decode_batch") {
                    require_exact(command, {"protocol_version", "type", "command_id", "steps"});
                    const auto & steps = command.at("steps");
                    if (
                        !steps.is_array()
                        || steps.empty()
                        || steps.size() > config.max_decode_batch
                    ) {
                        throw std::runtime_error("decode_steps_invalid");
                    }
                    struct Work {
                        NativeSequenceState * state;
                        json handle;
                        std::size_t token_budget;
                        std::vector<llama_token> token_ids;
                        std::string text_delta;
                    };
                    std::vector<Work> work;
                    std::vector<bool> seen(config.max_sequences, false);
                    std::size_t requested_tokens = 0;
                    for (const auto & step : steps) {
                        require_exact(step, {"handle", "token_budget"});
                        const auto token_budget = positive_size(
                            step.at("token_budget"),
                            "decode_budget_invalid"
                        );
                        if (token_budget > config.decode_quantum_tokens) {
                            throw std::runtime_error("decode_budget_invalid");
                        }
                        requested_tokens += token_budget;
                        const auto parsed = parse_public_handle(
                            step.at("handle"), config, model_id
                        );
                        if (!parsed.has_value() || parsed->first >= states.size()) {
                            throw std::runtime_error("stale_handle");
                        }
                        if (seen[parsed->first]) {
                            throw std::runtime_error("duplicate_sequence_step");
                        }
                        seen[parsed->first] = true;
                        auto * state = states[parsed->first].get();
                        if (
                            state == nullptr
                            || state->internal.generation != parsed->second
                            || state->terminal
                            || engine.snapshot(state->internal)->lifecycle
                                != SequenceLifecycle::decode
                        ) {
                            throw std::runtime_error("stale_handle");
                        }
                        work.push_back({state, step.at("handle"), token_budget, {}, {}});
                    }
                    if (requested_tokens > config.tick_token_budget) {
                        throw std::runtime_error("tick_token_budget_exceeded");
                    }

                    while (true) {
                        struct Active {
                            std::size_t work_index;
                            std::int32_t output_index;
                        };
                        std::vector<Active> active;
                        active.reserve(work.size());
                        Batch batch(static_cast<std::int32_t>(work.size()));
                        for (std::size_t index = 0; index < work.size(); ++index) {
                            auto & item = work[index];
                            if (
                                item.state->terminal
                                || item.token_ids.size() >= item.token_budget
                            ) {
                                continue;
                            }
                            const auto output_index = batch.value.n_tokens;
                            batch_add(
                                batch.value,
                                item.state->pending_token,
                                item.state->n_past,
                                item.state->internal.slot,
                                true
                            );
                            active.push_back({index, output_index});
                        }
                        if (active.empty()) break;

                        const auto started = std::chrono::steady_clock::now();
                        if (llama_decode(context.get(), batch.value) != 0) {
                            throw std::runtime_error("decode_failed");
                        }
                        llama_synchronize(context.get());
                        std::vector<llama_token> sampled_tokens(
                            active.size(),
                            LLAMA_TOKEN_NULL
                        );
                        std::vector<std::pair<std::size_t, std::future<llama_token>>>
                            cpu_samples;
                        cpu_samples.reserve(active.size());
                        for (std::size_t sample_index = 0;
                             sample_index < active.size();
                             ++sample_index) {
                            const auto & current = active[sample_index];
                            auto & state = *work[current.work_index].state;
                            if (state.backend_reasoning_sampler && !state.final_phase) {
                                sampled_tokens[sample_index] = llama_get_sampled_token_ith(
                                    context.get(),
                                    current.output_index
                                );
                                if (sampled_tokens[sample_index] == LLAMA_TOKEN_NULL) {
                                    throw std::runtime_error("backend_sampling_failed");
                                }
                                continue;
                            }
                            const auto * logits = llama_get_logits_ith(
                                context.get(),
                                current.output_index
                            );
                            if (!logits) throw std::runtime_error("sampling_logits_missing");
                            auto * sampler = state.final_phase
                                ? state.final_sampler.get()
                                : state.reasoning_sampler.get();
                            cpu_samples.emplace_back(
                                sample_index,
                                std::async(
                                    std::launch::async,
                                    [
                                        &state,
                                        logits,
                                        sampler,
                                        vocabulary_size,
                                        grammar_rejection = state.final_phase
                                    ]() {
                                        return sample_from_logits(
                                            sampler,
                                            logits,
                                            vocabulary_size,
                                            state.sampling_candidates,
                                            grammar_rejection
                                        );
                                    }
                                )
                            );
                        }
                        for (auto & [sample_index, future] : cpu_samples) {
                            sampled_tokens[sample_index] = future.get();
                        }
                        for (std::size_t sample_index = 0;
                             sample_index < active.size();
                             ++sample_index) {
                            const auto & current = active[sample_index];
                            auto & item = work[current.work_index];
                            auto & state = *item.state;
                            ++state.n_past;
                            const auto phase_before = state.phase->phase();
                            const auto token = sampled_tokens[sample_index];
                            item.token_ids.push_back(token);
                            const bool eog = llama_vocab_is_eog(vocab, token);
                            const auto sampled_at = std::chrono::steady_clock::now();
                            if (state.last_sample_at.has_value()) {
                                state.sample_itl_ms.push_back(
                                    std::chrono::duration<double, std::milli>(
                                        sampled_at - *state.last_sample_at
                                    ).count()
                                );
                            } else {
                                state.first_sample_ms =
                                    std::chrono::duration<double, std::milli>(
                                        sampled_at - state.opened_at
                                    ).count();
                            }
                            state.last_sample_at = sampled_at;
                            if (phase_before == ReasoningPhase::final && !eog) {
                                if (state.last_final_at.has_value()) {
                                    state.final_itl_ms.push_back(
                                        std::chrono::duration<double, std::milli>(
                                            sampled_at - *state.last_final_at
                                        ).count()
                                    );
                                } else {
                                    state.first_final_ms =
                                        std::chrono::duration<double, std::milli>(
                                            sampled_at - state.opened_at
                                        ).count();
                                }
                                state.last_final_at = sampled_at;
                            }
                            const auto source = state.final_phase
                                ? GrammarSampleSource::final_grammar
                                : GrammarSampleSource::unconstrained;
                            const auto transition = state.phase->consume(
                                token,
                                eog,
                                model_worker::grammar_acceptance_proven(eog, source)
                            );
                            if (
                                engine.advance_decode(state.internal, 1)
                                != SequenceOperationStatus::applied
                            ) {
                                throw std::runtime_error("sequence_state_invalid");
                            }
                            if (phase_before == ReasoningPhase::final && !eog) {
                                const auto bytes = token_piece(vocab, token);
                                state.final_bytes += bytes;
                                const auto appended = state.utf8.append(bytes);
                                if (!appended.valid) {
                                    state.failed = true;
                                    state.error_code = "invalid_utf8";
                                } else {
                                    item.text_delta += appended.completed;
                                }
                            }
                            if (transition.action == ControllerAction::activate_final_grammar) {
                                if (state.backend_reasoning_sampler) {
                                    if (!llama_set_sampler(
                                        context.get(),
                                        state.internal.slot,
                                        nullptr
                                    )) {
                                        throw std::runtime_error("backend_sampler_detach_failed");
                                    }
                                    state.backend_reasoning_sampler = false;
                                }
                                state.final_phase = true;
                            }
                            if (transition.action == ControllerAction::failed) {
                                state.failed = true;
                                state.error_code = transition.error;
                            }
                            const auto snapshot = engine.snapshot(state.internal);
                            if (!snapshot.has_value()) {
                                throw std::runtime_error("sequence_state_invalid");
                            }
                            if (
                                !state.failed
                                && transition.action != ControllerAction::completed
                                && snapshot->decoded_tokens == snapshot->output_token_budget
                            ) {
                                state.failed = true;
                                state.error_code = "total_budget_exhausted";
                            }
                            if (state.failed) {
                                state.terminal = true;
                                engine.finish(state.internal, SequenceFinishReason::failed);
                            } else if (transition.action == ControllerAction::completed) {
                                state.terminal = true;
                                if (!state.utf8.finish()) {
                                    state.failed = true;
                                    state.error_code = "incomplete_utf8";
                                    engine.finish(
                                        state.internal,
                                        SequenceFinishReason::failed
                                    );
                                } else {
                                    engine.finish(state.internal, SequenceFinishReason::stop);
                                }
                            } else {
                                state.pending_token = token;
                            }
                        }
                        const auto elapsed = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started
                        ).count();
                        for (const auto & current : active) {
                            work[current.work_index].state->generation_ms += elapsed;
                        }
                    }

                    auto outcomes = json::array();
                    for (auto & item : work) {
                        auto & state = *item.state;
                        json outcome{
                            {"handle", item.handle},
                            {"token_ids", item.token_ids},
                            {"text_delta", item.text_delta},
                        };
                        if (state.failed) {
                            outcome["status"] = "failed";
                            outcome["error_code"] = "protocol_violation:" + state.error_code;
                        } else if (state.terminal) {
                            outcome["status"] = "finished";
                            outcome["finish_reason"] = "stop";
                            outcome["completion"] = {
                                {"final_text", state.final_bytes},
                                {"prompt_tokens", state.prompt.size()},
                                {"reasoning_tokens", state.phase->reasoning_tokens()},
                                {"final_tokens", state.phase->final_tokens()},
                                {"sampled_tokens", state.phase->sampled_tokens()},
                                {"cached_prompt_tokens", state.cached_prompt_tokens},
                                {"cache_hit", state.cache_hit},
                                {"cache_match", state.cache_match},
                                {"session_id", state.session.has_value()
                                    ? json(state.session->session_id)
                                    : json(nullptr)},
                                {"session_parent_generation", (
                                    state.session.has_value()
                                    && state.session->parent_generation.has_value()
                                )
                                    ? json(*state.session->parent_generation)
                                    : json(nullptr)},
                                {"session_generation",
                                    state.committed_session_generation.has_value()
                                        ? json(*state.committed_session_generation)
                                        : json(nullptr)},
                                {"session_copy_on_write",
                                    state.session_copy_on_write},
                                {"prompt_decode_ms", state.prompt_decode_ms},
                                {"generation_ms", state.generation_ms},
                                {"first_sample_ms", state.first_sample_ms},
                                {"first_final_ms", state.first_final_ms},
                                {"sample_itl_ms", state.sample_itl_ms},
                                {"final_itl_ms", state.final_itl_ms},
                                {"grammar_engine", state.final_grammar_engine},
                            };
                        } else {
                            outcome["status"] = "progressed";
                        }
                        outcomes.push_back(std::move(outcome));
                    }
                    auto response = response_base("decode_completed", command_id);
                    response["outcomes"] = std::move(outcomes);
                    emit(response);
                    continue;
                }

                if (type == "release_sequence") {
                    require_exact(command, {"protocol_version", "type", "command_id", "handle"});
                    const auto parsed = parse_public_handle(
                        command.at("handle"), config, model_id
                    );
                    if (!parsed.has_value()) {
                        throw std::runtime_error("stale_handle");
                    }
                    model_worker::SequenceHandle internal{
                        static_cast<std::int32_t>(parsed->first),
                        parsed->second,
                        "",
                        "",
                    };
                    auto * state = states[parsed->first].get();
                    if (state != nullptr && state->internal.generation == parsed->second) {
                        internal = state->internal;
                        const auto snapshot = engine.snapshot(internal);
                        if (
                            snapshot.has_value()
                            && snapshot->lifecycle != SequenceLifecycle::terminal
                        ) {
                            engine.cancel(internal);
                        }
                    }
                    const auto memory_size = state == nullptr
                        ? 0
                        : llama_state_seq_get_size(context.get(), internal.slot);
                    if (state != nullptr) {
                        if (!llama_set_sampler(context.get(), internal.slot, nullptr)) {
                            throw std::runtime_error("backend_sampler_release_failed");
                        }
                        state->backend_reasoning_sampler = false;
                        if (!llama_memory_seq_rm(
                            llama_get_memory(context.get()),
                            internal.slot,
                            -1,
                            -1
                        )) {
                            throw std::runtime_error("sequence_memory_release_failed");
                        }
                    }
                    const auto released = engine.release(internal);
                    if (released.status == SequenceReleaseStatus::released) {
                        states[parsed->first].reset();
                    }
                    auto response = response_base("sequence_released", command_id);
                    response["handle"] = command.at("handle");
                    switch (released.status) {
                        case SequenceReleaseStatus::released:
                            response["status"] = "released";
                            response["released_bytes"] = memory_size;
                            break;
                        case SequenceReleaseStatus::already_released:
                            response["status"] = "already_released";
                            response["released_bytes"] = 0;
                            break;
                        case SequenceReleaseStatus::stale_handle:
                            response["status"] = "stale_handle";
                            response["released_bytes"] = 0;
                            break;
                        default:
                            response["status"] = "invalid_state";
                            response["released_bytes"] = 0;
                            break;
                    }
                    emit(response);
                    continue;
                }

                throw std::runtime_error("unknown_command");
            } catch (const std::exception & error) {
                emit_error(command_id, "invalid_command", error.what());
            }
        }

        for (std::size_t slot = 0; slot < states.size(); ++slot) {
            if (states[slot] != nullptr) {
                llama_set_sampler(
                    context.get(),
                    static_cast<llama_seq_id>(slot),
                    nullptr
                );
                states[slot]->backend_reasoning_sampler = false;
                llama_memory_seq_rm(
                    llama_get_memory(context.get()),
                    static_cast<llama_seq_id>(slot),
                    -1,
                    -1
                );
            }
        }
        context.reset();
        model.reset();
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "startup failed: " << error.what() << '\n';
        llama_backend_free();
        return 70;
    }
}
