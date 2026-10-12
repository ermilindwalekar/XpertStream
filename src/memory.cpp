#include "xpertstream/memory/fixed_tensor_buffer.hpp"

namespace xpertstream {

FixedTensorBufferPool::FixedTensorBufferPool(const StaticTensorShapeConfig& config)
    : config_(config),
      input_ids_(config.decode_step_len, 0),
      position_ids_(config.decode_step_len, 0),
      attention_mask_(config.max_seq_len, 0) {}

std::span<token_id_t> FixedTensorBufferPool::input_ids_buffer() noexcept {
    return input_ids_;
}

std::span<const token_id_t> FixedTensorBufferPool::input_ids_buffer() const noexcept {
    return input_ids_;
}

std::span<int64_t> FixedTensorBufferPool::position_ids_buffer() noexcept {
    return position_ids_;
}

std::span<const int64_t> FixedTensorBufferPool::position_ids_buffer() const noexcept {
    return position_ids_;
}

std::span<int64_t> FixedTensorBufferPool::attention_mask_buffer() noexcept {
    return attention_mask_;
}

std::span<const int64_t> FixedTensorBufferPool::attention_mask_buffer() const noexcept {
    return attention_mask_;
}

std::span<float> FixedTensorBufferPool::logits_buffer() noexcept {
    return logits_;
}

std::span<const float> FixedTensorBufferPool::logits_buffer() const noexcept {
    return logits_;
}

void FixedTensorBufferPool::allocate_logits_buffer(uint32_t vocab_size) {
    logits_.assign(vocab_size, 0.0f);
}

} // namespace xpertstream
