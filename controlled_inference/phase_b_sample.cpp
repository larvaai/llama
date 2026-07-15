#include "llama.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static std::string json_escape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else if (c < 0x80) {
                    out += static_cast<char>(c);
                } else {
                    size_t length = (c >= 0xC2 && c <= 0xDF) ? 2 :
                                    (c >= 0xE0 && c <= 0xEF) ? 3 :
                                    (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
                    bool valid = length > 0 && i + length <= s.size();
                    for (size_t j = 1; valid && j < length; ++j) {
                        unsigned char continuation = static_cast<unsigned char>(s[i + j]);
                        valid = continuation >= 0x80 && continuation <= 0xBF;
                    }
                    if (valid) {
                        out.append(s, i, length);
                        i += length - 1;
                    } else {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    }
                }
        }
    }
    return out;
}

static bool valid_utf8(const std::string & bytes) {
    const auto cont = [](unsigned char c) { return c >= 0x80 && c <= 0xBF; };
    for (size_t i = 0; i < bytes.size();) {
        unsigned char c = static_cast<unsigned char>(bytes[i]);
        if (c <= 0x7F) { ++i; continue; }
        if (c >= 0xC2 && c <= 0xDF) {
            if (i + 1 >= bytes.size() || !cont(static_cast<unsigned char>(bytes[i + 1]))) return false;
            i += 2; continue;
        }
        if (c >= 0xE0 && c <= 0xEF) {
            if (i + 2 >= bytes.size()) return false;
            unsigned char c1 = static_cast<unsigned char>(bytes[i + 1]);
            unsigned char c2 = static_cast<unsigned char>(bytes[i + 2]);
            if (!cont(c2) || (c == 0xE0 ? c1 < 0xA0 || c1 > 0xBF :
                              c == 0xED ? c1 < 0x80 || c1 > 0x9F : !cont(c1))) return false;
            i += 3; continue;
        }
        if (c >= 0xF0 && c <= 0xF4) {
            if (i + 3 >= bytes.size()) return false;
            unsigned char c1 = static_cast<unsigned char>(bytes[i + 1]);
            if ((c == 0xF0 ? c1 < 0x90 || c1 > 0xBF :
                 c == 0xF4 ? c1 < 0x80 || c1 > 0x8F : !cont(c1)) ||
                !cont(static_cast<unsigned char>(bytes[i + 2])) ||
                !cont(static_cast<unsigned char>(bytes[i + 3]))) return false;
            i += 4; continue;
        }
        return false;
    }
    return true;
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(32);
    int n = llama_token_to_piece(vocab, token, buf.data(), (int) buf.size(), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), (int) buf.size(), 0, true);
    }
    return n > 0 ? std::string(buf.data(), n) : std::string();
}

static std::string read_text_file(const char * path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: phase_b_sample MODEL.gguf OUTPUT.jsonl\n";
        return 2;
    }
    const char * model_path = argv[1];
    const char * output_path = argv[2];
    const std::string mode = argc >= 5 ? argv[4] : "observe";
    const bool controlled = mode == "grammar" || mode == "schema";
    const bool schema_mode = mode == "schema";
    constexpr llama_token THINK_START = 248068;
    const int reasoning_budget = argc >= 9 ? std::stoi(argv[8]) : 768;
    const int final_budget = argc >= 10 ? std::stoi(argv[9]) : 256;
    const int total_budget = argc >= 11 ? std::stoi(argv[10]) : 1024;
    const std::string cancel_file = argc >= 12 ? argv[11] : "-";
    const llama_token think_end_id = argc >= 13 ? std::stoi(argv[12]) : 248069;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc >= 4 ? std::stoi(argv[3]) : 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { std::cerr << "model load failed\n"; return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string system_prompt;
    std::string user_prompt;
    if (schema_mode) {
        if (argc < 8) { std::cerr << "schema mode requires grammar, system prompt, and user prompt files\n"; return 8; }
        system_prompt = read_text_file(argv[6]);
        user_prompt = read_text_file(argv[7]);
        if (system_prompt.empty() || user_prompt.empty()) { std::cerr << "cannot read prompt files\n"; return 8; }
    } else {
        system_prompt = controlled
            ? "Think first. In the final answer return exactly one JSON object with one integer field named result."
            : "Think first. Then answer with one short JSON object.";
        user_prompt = "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. The final result must be 2.";
    }
    llama_chat_message messages[] = {
        {"system", system_prompt.c_str()},
        {"user", user_prompt.c_str()},
    };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    int32_t needed = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (needed < 0) { std::cerr << "chat template failed\n"; return 4; }
    std::vector<char> prompt_buf(needed + 1);
    llama_chat_apply_template(tmpl, messages, 2, true, prompt_buf.data(), (int32_t) prompt_buf.size());
    std::string prompt(prompt_buf.data(), needed);

    int32_t n_prompt = -llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.data(), (int32_t) prompt.size(), tokens.data(), n_prompt, true, true) < 0) {
        std::cerr << "tokenize failed\n"; return 5;
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 1024;
    cp.n_ubatch = 512;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::cerr << "context init failed\n"; return 6; }

    auto sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * thinking_sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(thinking_sampler, llama_sampler_init_greedy());
    llama_sampler * final_sampler = nullptr;
    if (controlled) {
        std::string grammar_storage;
        if (schema_mode) {
            if (argc < 6) { std::cerr << "schema mode requires a grammar file\n"; return 8; }
            std::ifstream grammar_file(argv[5], std::ios::binary);
            if (!grammar_file) { std::cerr << "cannot open grammar file\n"; return 8; }
            grammar_storage.assign(std::istreambuf_iterator<char>(grammar_file), std::istreambuf_iterator<char>());
        } else {
            grammar_storage =
                "root ::= \"{\" \"\\\"\" \"result\" \"\\\"\" \":\" integer \"}\"\n"
                "integer ::= \"0\" | [1-9] [0-9]*\n";
        }
        final_sampler = llama_sampler_chain_init(sp);
        llama_sampler * grammar_sampler = llama_sampler_init_grammar(vocab, grammar_storage.c_str(), "root");
        if (!grammar_sampler) { std::cerr << "grammar init failed\n"; return 7; }
        llama_sampler_chain_add(final_sampler, grammar_sampler);
        llama_sampler_chain_add(final_sampler, llama_sampler_init_greedy());
    }
    llama_sampler * sampler = thinking_sampler;

    std::ofstream out(output_path, std::ios::binary);
    out << "{\"type\":\"meta\",\"prompt_tokens\":" << n_prompt
        << ",\"think_start_id\":" << THINK_START << ",\"think_end_id\":" << think_end_id
        << ",\"controlled\":" << (controlled ? "true" : "false")
        << ",\"reasoning_budget\":" << reasoning_budget << ",\"final_budget\":" << final_budget
        << ",\"total_budget\":" << total_budget << "}\n";

    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
    enum class Phase { START, THINKING, FINAL } phase = Phase::START;
    int index = 0;
    bool grammar_active = false;
    std::string final_bytes;
    std::string termination = "total_budget_exhausted";
    int thinking_tokens = 0;
    int final_tokens = 0;
    int sampled_tokens = 0;
    std::string thinking_stream_pending;
    std::string final_stream_pending;
    auto started = std::chrono::steady_clock::now();
    for (; index < total_budget; ++index) {
        if (cancel_file != "-" && std::filesystem::exists(cancel_file)) {
            termination = "cancelled";
            break;
        }
        if (llama_decode(ctx, batch) != 0) { std::cerr << "decode failed\n"; termination = "decode_failed"; break; }
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        ++sampled_tokens;
        std::string p = piece(vocab, token);
        std::string before = phase == Phase::START ? "START" : phase == Phase::THINKING ? "THINKING" : "FINAL";
        if (token == THINK_START) phase = Phase::THINKING;
        if (token == think_end_id) phase = Phase::FINAL;
        std::string after = phase == Phase::START ? "START" : phase == Phase::THINKING ? "THINKING" : "FINAL";
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        const bool piece_utf8_valid = valid_utf8(p);
        std::string stream_channel;
        std::string text_delta;
        const bool is_control = token == THINK_START || token == think_end_id || llama_vocab_is_eog(vocab, token);
        if (!is_control) {
            stream_channel = before == "FINAL" ? "final" : "thinking";
            std::string & pending = before == "FINAL" ? final_stream_pending : thinking_stream_pending;
            pending += p;
            if (valid_utf8(pending)) {
                text_delta = pending;
                pending.clear();
            }
        }
        out << "{\"type\":\"token\",\"index\":" << index << ",\"ms\":" << ms
            << ",\"token_id\":" << token
            << ",\"piece_display\":" << (piece_utf8_valid ? "\"" + json_escape(p) + "\"" : "null")
            << ",\"piece_utf8_valid\":" << (piece_utf8_valid ? "true" : "false")
            << ",\"stream_channel\":" << (stream_channel.empty() ? "null" : "\"" + stream_channel + "\"")
            << ",\"text_delta\":" << (text_delta.empty() ? "null" : "\"" + json_escape(text_delta) + "\"")
            << ",\"phase_before\":\"" << before << "\",\"phase_after\":\"" << after
            << "\",\"is_boundary\":" << (token == think_end_id ? "true" : "false")
            << ",\"grammar_active\":" << (grammar_active ? "true" : "false") << "}\n";
        out.flush();
        if (after == "THINKING" && token != THINK_START) ++thinking_tokens;
        if (before == "FINAL" && !llama_vocab_is_eog(vocab, token)) ++final_tokens;
        if (before == "FINAL" && !llama_vocab_is_eog(vocab, token)) final_bytes += p;
        if (llama_vocab_is_eog(vocab, token)) {
            termination = phase == Phase::FINAL ? "completed" : "missing_reasoning_boundary";
            break;
        }
        if (controlled && token == think_end_id) {
            sampler = final_sampler;
            grammar_active = true;
            out << "{\"type\":\"sampler_switch\",\"after_token_index\":" << index
                << ",\"from\":\"free_greedy\",\"to\":\"json_grammar_plus_greedy\"}\n";
            out.flush();
        }
        if (phase == Phase::THINKING && thinking_tokens >= reasoning_budget) {
            termination = "reasoning_budget_exhausted";
            break;
        }
        if (phase == Phase::FINAL && final_tokens >= final_budget) {
            termination = "final_budget_exhausted";
            break;
        }
        batch = llama_batch_get_one(&token, 1);
    }

    const bool final_utf8_valid = valid_utf8(final_bytes);
    out << "{\"type\":\"summary\",\"sampled_tokens\":" << sampled_tokens
        << ",\"final_phase\":\"" << (phase == Phase::FINAL ? "FINAL" : phase == Phase::THINKING ? "THINKING" : "START")
        << "\",\"grammar_activated\":" << (grammar_active ? "true" : "false")
        << ",\"termination\":\"" << termination << "\",\"thinking_tokens\":" << thinking_tokens
        << ",\"final_tokens\":" << final_tokens
        << ",\"final_utf8_valid\":" << (final_utf8_valid ? "true" : "false")
        << ",\"final_text\":";
    if (final_utf8_valid) out << "\"" << json_escape(final_bytes) << "\"";
    else out << "null";
    out << "}\n";
    llama_sampler_free(thinking_sampler);
    if (final_sampler) llama_sampler_free(final_sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    if (termination == "completed") return 0;
    if (termination == "cancelled") return 20;
    if (termination == "reasoning_budget_exhausted") return 21;
    if (termination == "final_budget_exhausted") return 22;
    if (termination == "missing_reasoning_boundary") return 23;
    if (termination == "total_budget_exhausted") return 24;
    return 25;
}
