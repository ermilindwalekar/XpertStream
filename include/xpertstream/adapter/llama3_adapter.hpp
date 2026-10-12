#pragma once

#include <string>
#include <vector>
#include <span>
#include <sstream>
#include "xpertstream/adapter/llm_model_adapter.hpp"

namespace xpertstream {

/**
 * @brief Concrete adapter for Llama 3 / 3.1 / 3.2 instruction-tuned models.
 * 
 * Formats standard OpenAI chat messages into the Llama 3 prompt template:
 *   <|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>
 *   <|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>
 *   <|start_header_id|>assistant<|end_header_id|>\n
 * And parses output streams for tool calls or terminal stop tokens.
 */
class Llama3Adapter final : public LLMModelAdapter {
public:
    explicit Llama3Adapter(seq_len_t max_seq_len = 2048, const std::string& model_name = "Meta-Llama-3.1-8B-Instruct") {
        metadata_.model_name = model_name;
        metadata_.model_family = "Llama-3";
        metadata_.vocab_size = 128256;
        metadata_.max_seq_len = max_seq_len;
        metadata_.num_layers = 32;
        metadata_.num_attention_heads = 32;
        metadata_.num_key_value_heads = 8;
        metadata_.hidden_size = 4096;

        metadata_.special_tokens.bos_token_id = 128000; // <|begin_of_text|>
        metadata_.special_tokens.eos_token_id = 128001; // <|end_of_text|>
        metadata_.special_tokens.pad_token_id = 128004;
        metadata_.special_tokens.stop_token_ids = {128001, 128008, 128009}; // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    }

    [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
        return metadata_;
    }

    [[nodiscard]] std::string format_chat_text(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools
    ) const {
        std::ostringstream oss;
        oss << "<|begin_of_text|>";

        // System prompt injection with tools
        bool has_system = false;
        for (const auto& msg : messages) {
            if (msg.role == Role::System) {
                has_system = true;
                oss << "<|start_header_id|>system<|end_header_id|>\n\n" << msg.content;
                if (!tools.empty()) {
                    oss << "\n\nAvailable tools:\n";
                    for (const auto& t : tools) {
                        oss << "- " << t.function.name << ": " << t.function.description
                            << " Parameters: " << t.function.parameters_schema_json << "\n";
                    }
                }
                oss << "<|eot_id|>";
                break;
            }
        }

        if (!has_system && !tools.empty()) {
            oss << "<|start_header_id|>system<|end_header_id|>\n\nAvailable tools:\n";
            for (const auto& t : tools) {
                oss << "- " << t.function.name << ": " << t.function.description
                    << " Parameters: " << t.function.parameters_schema_json << "\n";
            }
            oss << "<|eot_id|>";
        }

        // Conversation history
        for (const auto& msg : messages) {
            if (msg.role == Role::System) continue;

            if (msg.role == Role::User) {
                oss << "<|start_header_id|>user<|end_header_id|>\n\n" << msg.content << "<|eot_id|>";
            } else if (msg.role == Role::Assistant) {
                oss << "<|start_header_id|>assistant<|end_header_id|>\n\n" << msg.content;
                for (const auto& tc : msg.tool_calls) {
                    oss << "\n<|python_tag|>" << tc.function.name << "(" << tc.function.arguments_json << ")";
                }
                oss << "<|eot_id|>";
            } else if (msg.role == Role::Tool) {
                oss << "<|start_header_id|>ipython<|end_header_id|>\n\n" << msg.content << "<|eot_id|>";
            }
        }

        // Prompt tail ready for assistant response
        oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return oss.str();
    }

    [[nodiscard]] std::vector<token_id_t> encode_chat_prompt(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools
    ) const override {
        // High-level tokenization mapping
        // When connected with openvino_genai::Tokenizer or HuggingFace tokenizers library,
        // this calls tokenizer.encode(format_chat_text(messages, tools)).
        std::string prompt_str = format_chat_text(messages, tools);
        std::vector<token_id_t> dummy_tokens;
        dummy_tokens.reserve(prompt_str.size() / 4 + 4);
        dummy_tokens.push_back(metadata_.special_tokens.bos_token_id);
        // Simple mock token hashing for demonstration if no tokenizer runtime loaded
        for (char c : prompt_str) {
            dummy_tokens.push_back(static_cast<token_id_t>(static_cast<unsigned char>(c) + 256));
        }
        return dummy_tokens;
    }

    [[nodiscard]] std::string decode_token(token_id_t token) const override {
        if (token >= 256 && token < 512) {
            return std::string(1, static_cast<char>(token - 256));
        }
        return "";
    }

    [[nodiscard]] std::string decode_tokens(std::span<const token_id_t> tokens) const override {
        std::string result;
        for (auto t : tokens) {
            result += decode_token(t);
        }
        return result;
    }

    [[nodiscard]] GenerationOutput parse_generation_output(
        std::span<const token_id_t> generated_tokens,
        const std::string& raw_text
    ) const override {
        GenerationOutput out;
        out.text = raw_text.empty() ? decode_tokens(generated_tokens) : raw_text;

        // Check for tool calling syntax: e.g. <|python_tag|>tool_name(args)
        const std::string tag = "<|python_tag|>";
        auto pos = out.text.find(tag);
        if (pos != std::string::npos) {
            std::string call_str = out.text.substr(pos + tag.size());
            auto paren_open = call_str.find('(');
            auto paren_close = call_str.rfind(')');
            if (paren_open != std::string::npos && paren_close != std::string::npos) {
                std::string func_name = call_str.substr(0, paren_open);
                std::string args = call_str.substr(paren_open + 1, paren_close - paren_open - 1);

                ToolCall tc;
                tc.id = "call_" + std::to_string(generated_tokens.size());
                tc.function.name = func_name;
                tc.function.arguments_json = args;

                out.tool_calls.push_back(tc);
                out.finish_reason = FinishReason::ToolCalls;
                out.text = out.text.substr(0, pos);
                return out;
            }
        }

        out.finish_reason = FinishReason::Stop;
        return out;
    }

    [[nodiscard]] bool is_stop_token(token_id_t token) const noexcept override {
        for (auto stop_id : metadata_.special_tokens.stop_token_ids) {
            if (token == stop_id) return true;
        }
        return false;
    }

private:
    ModelMetadata metadata_;
};

} // namespace xpertstream
