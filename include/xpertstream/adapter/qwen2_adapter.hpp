#pragma once

#include <string>
#include <vector>
#include <span>
#include <sstream>
#include "xpertstream/adapter/llm_model_adapter.hpp"

namespace xpertstream {

/**
 * @brief Concrete adapter for Qwen 2.5 instruction-tuned models.
 * 
 * Formats standard OpenAI chat messages into the ChatML / Qwen prompt template:
 *   <|im_start|>system\n{content}<|im_end|>\n
 *   <|im_start|>user\n{content}<|im_end|>\n
 *   <|im_start|>assistant\n
 * And parses output streams for tool calls or terminal stop tokens.
 */
class Qwen2Adapter final : public LLMModelAdapter {
public:
    explicit Qwen2Adapter(seq_len_t max_seq_len = 2048, const std::string& model_name = "Qwen2.5-Coder-7B-Instruct") {
        metadata_.model_name = model_name;
        metadata_.model_family = "Qwen";
        metadata_.max_seq_len = max_seq_len;

        // Auto-configure architectural metadata based on model scale
        if (model_name.find("7B") != std::string::npos || model_name.find("7b") != std::string::npos) {
            metadata_.vocab_size = 152064;
            metadata_.num_layers = 28;
            metadata_.num_attention_heads = 28;
            metadata_.num_key_value_heads = 4;
            metadata_.hidden_size = 3584;
        } else if (model_name.find("1.5B") != std::string::npos || model_name.find("1.5b") != std::string::npos) {
            metadata_.vocab_size = 151936;
            metadata_.num_layers = 28;
            metadata_.num_attention_heads = 12;
            metadata_.num_key_value_heads = 2;
            metadata_.hidden_size = 1536;
        } else {
            // Default 0.5B
            metadata_.vocab_size = 151936;
            metadata_.num_layers = 24;
            metadata_.num_attention_heads = 14;
            metadata_.num_key_value_heads = 2;
            metadata_.hidden_size = 896;
        }

        metadata_.special_tokens.bos_token_id = 151643;
        metadata_.special_tokens.eos_token_id = 151645; // <|im_end|>
        metadata_.special_tokens.pad_token_id = 151643;
        metadata_.special_tokens.stop_token_ids = {151643, 151645}; // <|endoftext|>, <|im_end|>
    }

    [[nodiscard]] const ModelMetadata& metadata() const noexcept override {
        return metadata_;
    }

    [[nodiscard]] std::string format_chat_text(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools
    ) const {
        std::ostringstream oss;

        // System prompt injection with tools
        bool has_system = false;
        for (const auto& msg : messages) {
            if (msg.role == Role::System) {
                has_system = true;
                oss << "<|im_start|>system\n" << msg.content;
                if (!tools.empty()) {
                    oss << "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n"
                        << "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
                    for (const auto& t : tools) {
                        oss << "{\"type\": \"function\", \"function\": {\"name\": \"" << t.function.name
                            << "\", \"description\": \"" << t.function.description
                            << "\", \"parameters\": " << t.function.parameters_schema_json << "}}\n";
                    }
                    oss << "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                        << "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
                }
                oss << "<|im_end|>\n";
                break;
            }
        }

        if (!has_system && !tools.empty()) {
            oss << "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n<tools>\n";
            for (const auto& t : tools) {
                oss << "{\"type\": \"function\", \"function\": {\"name\": \"" << t.function.name
                    << "\", \"description\": \"" << t.function.description
                    << "\", \"parameters\": " << t.function.parameters_schema_json << "}}\n";
            }
            oss << "</tools><|im_end|>\n";
        }

        // Conversation history
        for (const auto& msg : messages) {
            if (msg.role == Role::System) continue;

            if (msg.role == Role::User) {
                oss << "<|im_start|>user\n" << msg.content << "<|im_end|>\n";
            } else if (msg.role == Role::Assistant) {
                oss << "<|im_start|>assistant\n" << msg.content;
                for (const auto& tc : msg.tool_calls) {
                    oss << "\n<tool_call>\n{\"name\": \"" << tc.function.name
                        << "\", \"arguments\": " << tc.function.arguments_json << "}\n</tool_call>";
                }
                oss << "<|im_end|>\n";
            } else if (msg.role == Role::Tool) {
                oss << "<|im_start|>user\n<tool_response>\n" << msg.content << "\n</tool_response><|im_end|>\n";
            }
        }

        // Prompt tail ready for assistant response
        oss << "<|im_start|>assistant\n";
        return oss.str();
    }

    [[nodiscard]] std::vector<token_id_t> encode_chat_prompt(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools
    ) const override {
        std::string prompt_str = format_chat_text(messages, tools);
        std::vector<token_id_t> dummy_tokens;
        dummy_tokens.reserve(prompt_str.size() / 4 + 4);
        dummy_tokens.push_back(metadata_.special_tokens.bos_token_id);
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

        // Check for Qwen tool call XML tag: <tool_call>...</tool_call>
        const std::string tag_open = "<tool_call>";
        const std::string tag_close = "</tool_call>";
        auto start_pos = out.text.find(tag_open);
        auto end_pos = out.text.find(tag_close);

        if (start_pos != std::string::npos && end_pos != std::string::npos && end_pos > start_pos) {
            std::string body = out.text.substr(start_pos + tag_open.size(), end_pos - start_pos - tag_open.size());
            
            // Extract "name": "..." and "arguments": ...
            auto name_pos = body.find("\"name\":");
            if (name_pos != std::string::npos) {
                auto first_quote = body.find('"', name_pos + 7);
                auto second_quote = (first_quote != std::string::npos) ? body.find('"', first_quote + 1) : std::string::npos;
                if (second_quote != std::string::npos) {
                    std::string func_name = body.substr(first_quote + 1, second_quote - first_quote - 1);
                    std::string args_json = "{}";
                    auto args_pos = body.find("\"arguments\":");
                    if (args_pos != std::string::npos) {
                        auto brace_open = body.find('{', args_pos);
                        auto brace_close = body.rfind('}');
                        if (brace_open != std::string::npos && brace_close != std::string::npos && brace_close > brace_open) {
                            args_json = body.substr(brace_open, brace_close - brace_open + 1);
                        }
                    }

                    ToolCall tc;
                    tc.id = "call_qwen_" + std::to_string(generated_tokens.size());
                    tc.function.name = func_name;
                    tc.function.arguments_json = args_json;

                    out.tool_calls.push_back(tc);
                    out.finish_reason = FinishReason::ToolCalls;
                    out.text = out.text.substr(0, start_pos);
                    return out;
                }
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
