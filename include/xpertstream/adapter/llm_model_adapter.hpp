#pragma once

#include <span>
#include <string>
#include <vector>
#include <memory>
#include "xpertstream/common/types.hpp"
#include "xpertstream/adapter/model_metadata.hpp"
#include "xpertstream/adapter/tool_definition.hpp"

namespace xpertstream {

/**
 * @brief Abstract base class for model-agnostic LLM adapters.
 * 
 * The adapter decouples high-level agentic semantic concepts (OpenAI-style chat history,
 * tool manifests, JSON schemas) from low-level tokenization and tensor operations.
 * The core engine interacts solely with this interface and never directly with
 * model-specific token formatting or hardware runtimes.
 */
class LLMModelAdapter {
public:
    virtual ~LLMModelAdapter() = default;

    /**
     * @brief Retrieve immutable metadata describing the target model architecture.
     */
    [[nodiscard]] virtual const ModelMetadata& metadata() const noexcept = 0;

    /**
     * @brief Format and tokenize standard OpenAI-style chat messages and tool definitions.
     * 
     * @param messages Vector of ChatMessage structs (System, User, Assistant, Tool).
     * @param tools Available tool definitions formatted with JSON Schema parameters.
     * @return Encoded token sequence ready for static-buffer tensor injection.
     */
    [[nodiscard]] virtual std::vector<token_id_t> encode_chat_prompt(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools
    ) const = 0;

    /**
     * @brief Convert a raw token ID into its string representation.
     */
    [[nodiscard]] virtual std::string decode_token(token_id_t token) const = 0;

    /**
     * @brief Batch decode a slice of token IDs into text.
     */
    [[nodiscard]] virtual std::string decode_tokens(std::span<const token_id_t> tokens) const = 0;

    /**
     * @brief Parse cumulative or terminal generated tokens into structured Text or ToolCalls.
     * 
     * Handles model-specific tool calling syntaxes (e.g. Llama 3 tool formatting,
     * Qwen XML/JSON tags, Hermes format) and maps them into uniform ToolCall models.
     * 
     * @param generated_tokens The output tokens produced by inference.
     * @param raw_text Pre-decoded text if already available, or empty string.
     * @return Structured GenerationOutput containing text and/or extracted ToolCalls.
     */
    [[nodiscard]] virtual GenerationOutput parse_generation_output(
        std::span<const token_id_t> generated_tokens,
        const std::string& raw_text = ""
    ) const = 0;

    /**
     * @brief Query whether a specific token signals generation termination.
     */
    [[nodiscard]] virtual bool is_stop_token(token_id_t token) const noexcept = 0;
};

} // namespace xpertstream
