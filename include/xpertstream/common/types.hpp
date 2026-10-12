#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include <chrono>

namespace xpertstream {

using token_id_t = int32_t;
using seq_len_t  = uint32_t;
using batch_idx_t = uint32_t;

enum class Role {
    System,
    User,
    Assistant,
    Tool
};

struct ToolFunctionCall {
    std::string name;
    std::string arguments_json;
};

struct ToolCall {
    std::string id;
    std::string type{"function"};
    ToolFunctionCall function;
};

struct ChatMessage {
    Role role;
    std::string content;
    std::optional<std::string> name;
    std::optional<std::string> tool_call_id;
    std::vector<ToolCall> tool_calls;
};

enum class FinishReason {
    Stop,
    LengthLimit,
    ToolCalls,
    ContentFilter,
    Error
};

struct ExecutionMetrics {
    std::chrono::microseconds time_to_first_token{0};
    std::chrono::microseconds total_compute_time{0};
    std::chrono::microseconds sampling_time{0};
    double tokens_per_second{0.0};
    seq_len_t prompt_tokens{0};
    seq_len_t generated_tokens{0};
    seq_len_t context_tokens_allocated{0};
};

struct GenerationOutput {
    std::string text;
    std::vector<ToolCall> tool_calls;
    FinishReason finish_reason{FinishReason::Stop};
    ExecutionMetrics metrics;
};

enum class TargetNPUArch {
    NPU_37XX,     // Meteor Lake / Arrow Lake
    NPU_40XX,     // Lunar Lake
    NPU_50XX,     // Panther Lake & Next-Gen
    FutureArch
};

} // namespace xpertstream
