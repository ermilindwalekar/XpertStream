#pragma once

#include <cstdint>
#include <vector>
#include <span>
#include <memory>
#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"

namespace xpertstream {

/**
 * @brief Zero-allocation contiguous buffer pool locked to static shapes.
 * 
 * Pre-allocates all input/output tensors at initialization time to avoid OS heap
 * allocations and page faults during real-time token generation loops.
 */
class FixedTensorBufferPool {
public:
    explicit FixedTensorBufferPool(const StaticTensorShapeConfig& config);
    ~FixedTensorBufferPool() = default;

    FixedTensorBufferPool(const FixedTensorBufferPool&) = delete;
    FixedTensorBufferPool& operator=(const FixedTensorBufferPool&) = delete;
    FixedTensorBufferPool(FixedTensorBufferPool&&) noexcept = default;
    FixedTensorBufferPool& operator=(FixedTensorBufferPool&&) noexcept = default;

    /**
     * @brief Access static input token buffer [1, 1].
     */
    [[nodiscard]] std::span<token_id_t> input_ids_buffer() noexcept;
    [[nodiscard]] std::span<const token_id_t> input_ids_buffer() const noexcept;

    /**
     * @brief Access static position IDs buffer [1, 1].
     */
    [[nodiscard]] std::span<int64_t> position_ids_buffer() noexcept;
    [[nodiscard]] std::span<const int64_t> position_ids_buffer() const noexcept;

    /**
     * @brief Access pre-allocated static attention mask buffer [1, max_seq_len].
     */
    [[nodiscard]] std::span<int64_t> attention_mask_buffer() noexcept;
    [[nodiscard]] std::span<const int64_t> attention_mask_buffer() const noexcept;

    /**
     * @brief Access static output logits buffer [1, 1, vocab_size].
     */
    [[nodiscard]] std::span<float> logits_buffer() noexcept;
    [[nodiscard]] std::span<const float> logits_buffer() const noexcept;

    /**
     * @brief Set vocabulary size and allocate output buffer accordingly.
     */
    void allocate_logits_buffer(uint32_t vocab_size);

    [[nodiscard]] const StaticTensorShapeConfig& config() const noexcept { return config_; }

private:
    StaticTensorShapeConfig config_;
    std::vector<token_id_t> input_ids_;
    std::vector<int64_t> position_ids_;
    std::vector<int64_t> attention_mask_;
    std::vector<float> logits_;
};

} // namespace xpertstream
