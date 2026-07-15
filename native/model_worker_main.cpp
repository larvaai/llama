#include "llama.h"
#include "generation_safety.h"
#include "pending_cancel_registry.h"
#include "reasoning_phase_controller.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using model_worker::ControllerAction;
using model_worker::GrammarSampleSource;
using model_worker::PendingCancelRegistry;
using model_worker::ReasoningPhase;
using model_worker::ReasoningPhaseController;
using model_worker::Utf8Accumulator;

namespace {
constexpr const char * IPC_VERSION = "model-worker-ipc.v1";

struct ModelDeleter { void operator()(llama_model * p) const { if (p) llama_model_free(p); } };
struct ContextDeleter { void operator()(llama_context * p) const { if (p) llama_free(p); } };
struct SamplerDeleter { void operator()(llama_sampler * p) const { if (p) llama_sampler_free(p); } };
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

struct SharedControl {
    std::mutex mutex;
    std::condition_variable changed;
    std::queue<json> jobs;
    PendingCancelRegistry pending_cancels;
    std::string active_request;
    std::string active_attempt;
    std::atomic_bool cancel{false};
    std::atomic_bool shutdown{false};
};

json read_json_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open manifest");
    return json::parse(stream, nullptr, true, true);
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    int32_t count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, parse_special);
    if (count == 0) return {};
    if (count > 0) throw std::runtime_error("tokenizer size query unexpectedly succeeded");
    std::vector<llama_token> result(static_cast<std::size_t>(-count));
    count = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), result.data(), static_cast<int32_t>(result.size()), add_special, parse_special);
    if (count < 0) throw std::runtime_error("tokenization failed");
    result.resize(static_cast<std::size_t>(count));
    return result;
}

std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(32);
    int32_t count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    if (count < 0) {
        buffer.resize(static_cast<std::size_t>(-count));
        count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
    }
    return count > 0 ? std::string(buffer.data(), static_cast<std::size_t>(count)) : std::string{};
}

void emit(const std::string & type, const std::string & request_id, const std::string & attempt_id,
          std::uint64_t sequence, const json & payload = json::object()) {
    json frame = payload;
    frame["protocol_version"] = IPC_VERSION; frame["type"] = type;
    frame["request_id"] = request_id; frame["attempt_id"] = attempt_id; frame["sequence"] = sequence;
    std::cout << frame.dump(-1, ' ', false, json::error_handler_t::replace) << '\n' << std::flush;
}

void reader_loop(SharedControl & control) {
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            json frame = json::parse(line, nullptr, true, true);
            if (frame.value("protocol_version", "") != IPC_VERSION) continue;
            const auto type = frame.value("type", "");
            if (type == "shutdown") { control.shutdown = true; control.cancel = true; control.changed.notify_all(); return; }
            if (type == "cancel") {
                std::lock_guard lock(control.mutex);
                const auto key = std::make_pair(
                    frame.value("request_id", ""),
                    frame.value("attempt_id", "")
                );
                if (key.first.empty() || key.second.empty()) continue;
                if (key.first == control.active_request && key.second == control.active_attempt) {
                    control.cancel = true;
                } else {
                    control.pending_cancels.add(key);
                }
                continue;
            }
            if (type == "generate") {
                std::lock_guard lock(control.mutex); control.jobs.push(std::move(frame)); control.changed.notify_one();
            }
        } catch (...) { /* malformed control input never crashes the resident worker */ }
    }
    control.shutdown = true; control.cancel = true; control.changed.notify_all();
}

json generate(llama_model * model, const llama_vocab * vocab, const json & manifest, const json & envelope,
              SharedControl & control, std::uint64_t & sequence) {
    const auto request_started = std::chrono::steady_clock::now();
    const json & request = envelope.at("request");
    const json & limits = request.at("limits");
    const json & model_messages = envelope.at("model_messages");
    const auto context_config = manifest.at("context");
    std::vector<std::string> roles, contents;
    std::vector<llama_chat_message> messages;
    for (const auto & item : model_messages) { roles.push_back(item.at("role")); contents.push_back(item.at("content")); }
    messages.reserve(roles.size());
    for (std::size_t i = 0; i < roles.size(); ++i) messages.push_back({roles[i].c_str(), contents[i].c_str()});
    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) throw std::runtime_error("chat_template_missing");
    const int32_t needed = llama_chat_apply_template(chat_template, messages.data(), messages.size(), true, nullptr, 0);
    if (needed <= 0) throw std::runtime_error("chat_template_failed");
    std::vector<char> prompt_buffer(static_cast<std::size_t>(needed) + 1);
    const int32_t written = llama_chat_apply_template(chat_template, messages.data(), messages.size(), true, prompt_buffer.data(), prompt_buffer.size());
    if (written != needed) throw std::runtime_error("chat_template_failed");
    auto prompt = tokenize(vocab, std::string(prompt_buffer.data(), written), true, true);
    const int32_t n_ctx = context_config.at("n_ctx");
    const int32_t n_batch = context_config.at("n_batch");
    const int32_t reserve = limits.at("total_tokens");
    if (static_cast<int64_t>(prompt.size()) + reserve + 8 > n_ctx) throw std::runtime_error("context_overflow");

    auto cp = llama_context_default_params(); cp.n_ctx = n_ctx; cp.n_batch = n_batch; cp.n_ubatch = context_config.at("n_ubatch"); cp.no_perf = true;
    ContextPtr context(llama_init_from_model(model, cp));
    if (!context) throw std::runtime_error("context_init_failed");
    std::size_t prompt_decoded = 0;
    for (std::size_t offset = 0; offset < prompt.size(); offset += n_batch) {
        if (control.cancel) throw std::runtime_error("cancelled");
        const auto count = std::min<std::size_t>(n_batch, prompt.size() - offset);
        auto batch = llama_batch_get_one(prompt.data() + offset, static_cast<int32_t>(count));
        if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("decode_failed");
        prompt_decoded += count;
        emit("progress", control.active_request, control.active_attempt, sequence++, {{"phase", "prompt_decode"}, {"tokens", prompt_decoded}});
    }
    const auto prompt_finished = std::chrono::steady_clock::now();

    const auto start = tokenize(vocab, manifest["reasoning"]["start_text"], false, true);
    const auto end = tokenize(vocab, manifest["reasoning"]["end_text"], false, true);
    ReasoningPhaseController phase(start, end, limits["reasoning_tokens"], limits["final_tokens"], limits["total_tokens"], manifest["reasoning"].value("require_start", true));
    auto sp = llama_sampler_chain_default_params(); sp.no_perf = true;
    SamplerPtr reasoning(llama_sampler_chain_init(sp)); llama_sampler_chain_add(reasoning.get(), llama_sampler_init_greedy());
    SamplerPtr final(llama_sampler_chain_init(sp));
    llama_sampler * grammar = llama_sampler_init_grammar(vocab, envelope.at("grammar").get_ref<const std::string &>().c_str(), "root");
    if (!reasoning || !final || !grammar) throw std::runtime_error("sampler_init_failed");
    llama_sampler_chain_add(final.get(), grammar); llama_sampler_chain_add(final.get(), llama_sampler_init_greedy());
    llama_sampler * active = reasoning.get();
    std::string final_bytes;
    Utf8Accumulator utf8;
    llama_token last = prompt.back();
    bool prompt_already_decoded = true;
    auto last_heartbeat = std::chrono::steady_clock::now();
    while (phase.sampled_tokens() < static_cast<std::size_t>(limits.at("total_tokens").get<int>())) {
        if (control.cancel) throw std::runtime_error("cancelled");
        if (!prompt_already_decoded) {
            auto one = llama_batch_get_one(&last, 1);
            if (llama_decode(context.get(), one) != 0) throw std::runtime_error("decode_failed");
        }
        prompt_already_decoded = false;
        const auto before = phase.phase();
        llama_sampler * sampled_with = active;
        const llama_token token = llama_sampler_sample(sampled_with, context.get(), -1);
        const bool eog = llama_vocab_is_eog(vocab, token);
        const auto source = sampled_with == final.get()
            ? GrammarSampleSource::final_grammar
            : GrammarSampleSource::unconstrained;
        const auto transition = phase.consume(
            token,
            eog,
            model_worker::grammar_acceptance_proven(eog, source)
        );
        if (before == ReasoningPhase::final && !eog) {
            const auto bytes = piece(vocab, token);
            final_bytes += bytes;
            const auto appended = utf8.append(bytes);
            if (!appended.valid) throw std::runtime_error("protocol_violation:invalid_utf8");
            if (!appended.completed.empty()) {
                emit("final_delta", control.active_request, control.active_attempt, sequence++, {{"delta", appended.completed}});
            }
        }
        if (transition.action == ControllerAction::activate_final_grammar) {
            active = final.get(); emit("phase", control.active_request, control.active_attempt, sequence++, {{"phase", "final"}});
        }
        if (transition.action == ControllerAction::failed) throw std::runtime_error("protocol_violation:" + transition.error);
        if (transition.action == ControllerAction::completed) break;
        if (phase.sampled_tokens() % 16 == 0) {
            emit("progress", control.active_request, control.active_attempt, sequence++, {{"phase", before == ReasoningPhase::final ? "final" : "reasoning"}, {"tokens", phase.sampled_tokens()}});
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_heartbeat >= std::chrono::seconds(1)) {
            emit("heartbeat", control.active_request, control.active_attempt, sequence++, {{"sampled_tokens", phase.sampled_tokens()}});
            last_heartbeat = now;
        }
        last = token;
    }
    if (phase.phase() != ReasoningPhase::done || !utf8.finish()) throw std::runtime_error("protocol_violation:incomplete_final");
    const auto generation_finished = std::chrono::steady_clock::now();
    const auto prompt_ms = std::chrono::duration<double, std::milli>(prompt_finished - request_started).count();
    const auto generation_ms = std::chrono::duration<double, std::milli>(generation_finished - prompt_finished).count();
    return {{"final_text", final_bytes}, {"usage", {{"prompt_tokens", prompt.size()}, {"reasoning_tokens", phase.reasoning_tokens()}, {"final_tokens", phase.final_tokens()}, {"sampled_tokens", phase.sampled_tokens()}, {"context_limit", n_ctx}, {"context_headroom", n_ctx - prompt.size() - reserve - 8}}}, {"timing", {{"prompt_decode_ms", prompt_ms}, {"generation_ms", generation_ms}}}};
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2) { std::cerr << "usage: model-worker-native <verified-model-manifest>\n"; return 64; }
    try {
        const json manifest = read_json_file(argv[1]);
        if (manifest.at("manifest_version") != "model-manifest.v1" || manifest.at("runtime_build") != "b10012") throw std::runtime_error("manifest_version_mismatch");
        const auto & reasoning = manifest.at("reasoning");
        if (reasoning.value("mode", "") != "required_marker_sequence") throw std::runtime_error("reasoning_mode_unsupported");
        if (!reasoning.contains("require_start") ||
            !reasoning.at("require_start").is_boolean() ||
            !reasoning.at("require_start").get<bool>()) {
            throw std::runtime_error("reasoning_require_start_invalid");
        }
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = manifest.at("gpu").at("layers");
        ModelPtr model(llama_model_load_from_file(manifest.at("gguf_path").get_ref<const std::string &>().c_str(), mp));
        if (!model) throw std::runtime_error("model_load_failed");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const auto start = tokenize(vocab, manifest["reasoning"]["start_text"], false, true);
        const auto end = tokenize(vocab, manifest["reasoning"]["end_text"], false, true);
        if (start.empty() || end.empty() || start == end || !llama_model_chat_template(model.get(), nullptr) || llama_model_n_ctx_train(model.get()) < manifest["context"]["n_ctx"].get<int>()) throw std::runtime_error("capability_verification_failed");
        std::cout << json{{"protocol_version", IPC_VERSION}, {"type", "ready"}, {"sequence", 0}, {"model_id", manifest["id"]}}.dump() << '\n' << std::flush;
        SharedControl control; std::thread reader(reader_loop, std::ref(control));
        while (!control.shutdown) {
            json frame;
            {
                std::unique_lock lock(control.mutex); control.changed.wait(lock, [&]{ return control.shutdown || !control.jobs.empty(); });
                if (control.shutdown) break; frame = std::move(control.jobs.front()); control.jobs.pop();
                control.active_request = frame.value("request_id", ""); control.active_attempt = frame.value("attempt_id", "");
                const auto key = std::make_pair(control.active_request, control.active_attempt);
                control.cancel = control.pending_cancels.consume(key);
            }
            std::uint64_t sequence = 0;
            emit("started", control.active_request, control.active_attempt, sequence++);
            try {
                json result = generate(model.get(), vocab, manifest, frame.at("request"), control, sequence);
                emit("completed", control.active_request, control.active_attempt, sequence++, result);
            } catch (const std::exception & exc) {
                const std::string message = exc.what();
                const std::string code = message == "cancelled" ? "cancelled" : message == "context_overflow" ? "context_overflow" : message.rfind("protocol_violation", 0) == 0 ? "protocol_violation" : "decode_failed";
                emit("failed", control.active_request, control.active_attempt, sequence++, {{"error", code}, {"detail", message}});
            }
            std::lock_guard lock(control.mutex); control.active_request.clear(); control.active_attempt.clear();
        }
        control.shutdown = true; control.changed.notify_all(); if (reader.joinable()) reader.join();
        model.reset(); llama_backend_free(); return 0;
    } catch (const std::exception & exc) { std::cerr << "startup failed: " << exc.what() << '\n'; llama_backend_free(); return 70; }
}
