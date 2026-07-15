#include "llama.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string json_escape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
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
                    const size_t length = (c >= 0xC2 && c <= 0xDF) ? 2 :
                                          (c >= 0xE0 && c <= 0xEF) ? 3 :
                                          (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
                    bool valid = length > 0 && i + length <= s.size();
                    for (size_t j = 1; valid && j < length; ++j) {
                        const unsigned char continuation = static_cast<unsigned char>(s[i + j]);
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
        const unsigned char c = static_cast<unsigned char>(bytes[i]);
        if (c <= 0x7F) { ++i; continue; }
        if (c >= 0xC2 && c <= 0xDF) {
            if (i + 1 >= bytes.size() || !cont(static_cast<unsigned char>(bytes[i + 1]))) return false;
            i += 2; continue;
        }
        if (c >= 0xE0 && c <= 0xEF) {
            if (i + 2 >= bytes.size()) return false;
            const unsigned char c1 = static_cast<unsigned char>(bytes[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(bytes[i + 2]);
            if (!cont(c2) || (c == 0xE0 ? c1 < 0xA0 || c1 > 0xBF :
                              c == 0xED ? c1 < 0x80 || c1 > 0x9F : !cont(c1))) return false;
            i += 3; continue;
        }
        if (c >= 0xF0 && c <= 0xF4) {
            if (i + 3 >= bytes.size()) return false;
            const unsigned char c1 = static_cast<unsigned char>(bytes[i + 1]);
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
    int n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(-n);
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int>(buf.size()), 0, true);
    }
    return n > 0 ? std::string(buf.data(), n) : std::string();
}

static std::string read_text_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

static std::vector<std::string> split_tabs(const std::string & line) {
    std::vector<std::string> fields;
    size_t start = 0;
    for (;;) {
        const size_t at = line.find('\t', start);
        fields.push_back(line.substr(start, at == std::string::npos ? at : at - start));
        if (at == std::string::npos) break;
        start = at + 1;
    }
    return fields;
}

struct Request {
    std::string id;
    std::string output_path;
    std::string grammar_path;
    std::string system_path;
    std::string user_path;
    int reasoning_budget = 768;
    int final_budget = 256;
    int total_budget = 1024;
    std::string cancel_path;
    llama_token think_end_id = 248069;
};

struct Result {
    int exit_code = 25;
    std::string termination = "internal_error";
};

static Result generate_one(
    llama_model * model,
    const llama_vocab * vocab,
    const Request & req,
    int request_ordinal) {
    Result result;
    const std::string system_prompt = read_text_file(req.system_path);
    const std::string user_prompt = read_text_file(req.user_path);
    const std::string grammar = read_text_file(req.grammar_path);
    if (system_prompt.empty() || user_prompt.empty() || grammar.empty()) {
        result.termination = "input_file_error";
        return result;
    }

    llama_chat_message messages[] = {
        {"system", system_prompt.c_str()},
        {"user", user_prompt.c_str()},
    };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const int32_t needed = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (needed < 0) { result.termination = "chat_template_failed"; return result; }
    std::vector<char> prompt_buf(needed + 1);
    llama_chat_apply_template(tmpl, messages, 2, true, prompt_buf.data(), static_cast<int32_t>(prompt_buf.size()));
    const std::string prompt(prompt_buf.data(), needed);

    const int32_t n_prompt = -llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), nullptr, 0, true, true);
    if (n_prompt <= 0) { result.termination = "tokenize_failed"; return result; }
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), n_prompt, true, true) < 0) {
        result.termination = "tokenize_failed"; return result;
    }

    // A fresh context is the isolation boundary. The model object and its GPU weights stay alive.
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = 1024;
    cp.n_ubatch = 512;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { result.termination = "context_init_failed"; return result; }

    auto sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    llama_sampler * thinking_sampler = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(thinking_sampler, llama_sampler_init_greedy());
    llama_sampler * final_sampler = llama_sampler_chain_init(sp);
    llama_sampler * grammar_sampler = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
    if (!grammar_sampler) {
        llama_sampler_free(thinking_sampler);
        llama_sampler_free(final_sampler);
        llama_free(ctx);
        result.termination = "grammar_init_failed";
        return result;
    }
    llama_sampler_chain_add(final_sampler, grammar_sampler);
    llama_sampler_chain_add(final_sampler, llama_sampler_init_greedy());
    llama_sampler * sampler = thinking_sampler;

    std::ofstream out(req.output_path, std::ios::binary);
    if (!out) {
        llama_sampler_free(thinking_sampler);
        llama_sampler_free(final_sampler);
        llama_free(ctx);
        result.termination = "output_open_failed";
        return result;
    }
    constexpr llama_token THINK_START = 248068;
    out << "{\"type\":\"meta\",\"request_id\":\"" << json_escape(req.id)
        << "\",\"request_ordinal\":" << request_ordinal
        << ",\"model_load_count\":1,\"context_fresh\":true,\"prompt_tokens\":" << n_prompt
        << ",\"think_start_id\":" << THINK_START << ",\"think_end_id\":" << req.think_end_id
        << ",\"controlled\":true,\"reasoning_budget\":" << req.reasoning_budget
        << ",\"final_budget\":" << req.final_budget << ",\"total_budget\":" << req.total_budget << "}\n";
    out.flush();

    llama_batch batch = llama_batch_get_one(tokens.data(), n_prompt);
    enum class Phase { START, THINKING, FINAL } phase = Phase::START;
    bool grammar_active = false;
    std::string final_bytes;
    std::string termination = "total_budget_exhausted";
    int thinking_tokens = 0;
    int final_tokens = 0;
    int sampled_tokens = 0;
    std::string thinking_stream_pending;
    std::string final_stream_pending;
    const auto started = std::chrono::steady_clock::now();
    for (int index = 0; index < req.total_budget; ++index) {
        if (req.cancel_path != "-" && std::filesystem::exists(req.cancel_path)) {
            termination = "cancelled";
            break;
        }
        if (llama_decode(ctx, batch) != 0) { termination = "decode_failed"; break; }
        llama_token token = llama_sampler_sample(sampler, ctx, -1);
        ++sampled_tokens;
        const std::string p = piece(vocab, token);
        const std::string before = phase == Phase::START ? "START" : phase == Phase::THINKING ? "THINKING" : "FINAL";
        if (token == THINK_START) phase = Phase::THINKING;
        if (token == req.think_end_id) phase = Phase::FINAL;
        const std::string after = phase == Phase::START ? "START" : phase == Phase::THINKING ? "THINKING" : "FINAL";
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        const bool piece_utf8_valid = valid_utf8(p);
        std::string stream_channel;
        std::string text_delta;
        const bool is_control = token == THINK_START || token == req.think_end_id || llama_vocab_is_eog(vocab, token);
        if (!is_control) {
            stream_channel = before == "FINAL" ? "final" : "thinking";
            std::string & pending = before == "FINAL" ? final_stream_pending : thinking_stream_pending;
            pending += p;
            if (valid_utf8(pending)) { text_delta = pending; pending.clear(); }
        }
        out << "{\"type\":\"token\",\"index\":" << index << ",\"ms\":" << ms
            << ",\"token_id\":" << token
            << ",\"piece_display\":" << (piece_utf8_valid ? "\"" + json_escape(p) + "\"" : "null")
            << ",\"piece_utf8_valid\":" << (piece_utf8_valid ? "true" : "false")
            << ",\"stream_channel\":" << (stream_channel.empty() ? "null" : "\"" + stream_channel + "\"")
            << ",\"text_delta\":" << (text_delta.empty() ? "null" : "\"" + json_escape(text_delta) + "\"")
            << ",\"phase_before\":\"" << before << "\",\"phase_after\":\"" << after
            << "\",\"is_boundary\":" << (token == req.think_end_id ? "true" : "false")
            << ",\"grammar_active\":" << (grammar_active ? "true" : "false") << "}\n";
        out.flush();
        if (after == "THINKING" && token != THINK_START) ++thinking_tokens;
        if (before == "FINAL" && !llama_vocab_is_eog(vocab, token)) { ++final_tokens; final_bytes += p; }
        if (llama_vocab_is_eog(vocab, token)) {
            termination = phase == Phase::FINAL ? "completed" : "missing_reasoning_boundary";
            break;
        }
        if (token == req.think_end_id) {
            sampler = final_sampler;
            grammar_active = true;
            out << "{\"type\":\"sampler_switch\",\"after_token_index\":" << index
                << ",\"from\":\"free_greedy\",\"to\":\"json_grammar_plus_greedy\"}\n";
            out.flush();
        }
        if (phase == Phase::THINKING && thinking_tokens >= req.reasoning_budget) {
            termination = "reasoning_budget_exhausted"; break;
        }
        if (phase == Phase::FINAL && final_tokens >= req.final_budget) {
            termination = "final_budget_exhausted"; break;
        }
        batch = llama_batch_get_one(&token, 1);
    }

    const bool final_utf8_valid = valid_utf8(final_bytes);
    out << "{\"type\":\"summary\",\"request_id\":\"" << json_escape(req.id)
        << "\",\"request_ordinal\":" << request_ordinal << ",\"model_load_count\":1,\"context_fresh\":true"
        << ",\"sampled_tokens\":" << sampled_tokens
        << ",\"final_phase\":\"" << (phase == Phase::FINAL ? "FINAL" : phase == Phase::THINKING ? "THINKING" : "START")
        << "\",\"grammar_activated\":" << (grammar_active ? "true" : "false")
        << ",\"termination\":\"" << termination << "\",\"thinking_tokens\":" << thinking_tokens
        << ",\"final_tokens\":" << final_tokens << ",\"final_utf8_valid\":" << (final_utf8_valid ? "true" : "false")
        << ",\"final_text\":";
    if (final_utf8_valid) out << "\"" << json_escape(final_bytes) << "\"";
    else out << "null";
    out << "}\n";
    out.flush();

    llama_sampler_free(thinking_sampler);
    llama_sampler_free(final_sampler);
    llama_free(ctx);

    result.termination = termination;
    result.exit_code = termination == "completed" ? 0 :
                       termination == "cancelled" ? 20 :
                       termination == "reasoning_budget_exhausted" ? 21 :
                       termination == "final_budget_exhausted" ? 22 :
                       termination == "missing_reasoning_boundary" ? 23 :
                       termination == "total_budget_exhausted" ? 24 : 25;
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: persistent_worker MODEL.gguf [GPU_LAYERS]\n";
        return 2;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = argc >= 3 ? std::stoi(argv[2]) : 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::cerr << "model load failed\n"; llama_backend_free(); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::cout << "{\"type\":\"worker_ready\",\"model_load_count\":1}\n" << std::flush;

    int request_ordinal = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "shutdown") break;
        const auto fields = split_tabs(line);
        if (fields.size() != 10) {
            std::cout << "{\"type\":\"worker_protocol_error\",\"detail\":\"expected 10 fields\"}\n" << std::flush;
            continue;
        }
        Request req;
        req.id = fields[0];
        req.output_path = fields[1];
        req.grammar_path = fields[2];
        req.system_path = fields[3];
        req.user_path = fields[4];
        try {
            req.reasoning_budget = std::stoi(fields[5]);
            req.final_budget = std::stoi(fields[6]);
            req.total_budget = std::stoi(fields[7]);
            req.cancel_path = fields[8];
            req.think_end_id = std::stoi(fields[9]);
        } catch (...) {
            std::cout << "{\"type\":\"worker_protocol_error\",\"request_id\":\"" << json_escape(req.id)
                      << "\",\"detail\":\"invalid numeric field\"}\n" << std::flush;
            continue;
        }
        ++request_ordinal;
        const Result result = generate_one(model, vocab, req, request_ordinal);
        std::cout << "{\"type\":\"request_done\",\"request_id\":\"" << json_escape(req.id)
                  << "\",\"request_ordinal\":" << request_ordinal << ",\"model_load_count\":1"
                  << ",\"context_fresh\":true,\"exit_code\":" << result.exit_code
                  << ",\"termination\":\"" << json_escape(result.termination) << "\"}\n" << std::flush;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
