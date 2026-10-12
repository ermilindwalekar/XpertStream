#pragma once

#include <cstdint>
#include <string>
#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"

namespace xpertstream {

/**
 * @brief High-performance tracker for OpenVINO stateful KV-cache nodes (ReadValue/Assign).
 * 
 * OpenVINO compiles stateful LLMs with internal memory representations via ReadValue/Assign
 * layers. This class tracks context progression, coordinates mask updating, and enforces
 * the hard sequence boundary to ensure NPU execution never falls back to CPU.
 */
class StatefulKVCacheTracker {
public:
    explicit StatefulKVCacheTracker(
        const StaticTensorShapeConfig& config,
        TargetNPUArch target_arch = TargetNPUArch::NPU_40XX
    ) : config_(config), target_arch_(target_arch), current_seq_len_(0) {}

    /**
     * @brief Reset internal KV-cache state and clear position counter.
     */
    void reset() noexcept {
        current_seq_len_ = 0;
    }

    /**
     * @brief Validate whether incoming token step exceeds compiled maximum sequence limit.
     * 
     * @param incoming_tokens Number of tokens to append.
     * @param phase Diagnostic tag ("prefill" or "decode").
     * @throws SequenceOverflowException if current_seq_len + incoming_tokens > max_seq_len.
     */
    void validate_sequence_boundary(seq_len_t incoming_tokens, const char* phase) const {
        if (current_seq_len_ + incoming_tokens > config_.max_seq_len) {
            SequenceOverflowException::DiagnosticMetrics metrics{
                .current_seq_len = current_seq_len_,
                .max_seq_len = config_.max_seq_len,
                .requested_additional_tokens = incoming_tokens,
                .target_arch = target_arch_,
                .execution_phase = phase
            };
            throw SequenceOverflowException(metrics);
        }
    }

    /**
     * @brief Advance sequence cursor by designated token count after successful inference step.
     */
    void advance(seq_len_t token_count) noexcept {
        current_seq_len_ += token_count;
    }

    [[nodiscard]] seq_len_t current_seq_len() const noexcept {
        return current_seq_len_;
    }

    [[nodiscard]] seq_len_t max_seq_len() const noexcept {
        return config_.max_seq_len;
    }

    [[nodiscard]] seq_len_t remaining_capacity() const noexcept {
        return (config_.max_seq_len > current_seq_len_)
            ? (config_.max_seq_len - current_seq_len_)
            : 0;
    }

    [[nodiscard]] TargetNPUArch target_arch() const noexcept {
        return target_arch_;
    }

private:
    StaticTensorShapeConfig config_;
    TargetNPUArch target_arch_;
    seq_len_t current_seq_len_{0};
};

} // namespace xpertstream
