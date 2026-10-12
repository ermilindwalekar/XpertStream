#pragma once

#include <cstdint>
#include <array>
#include <span>
#include "xpertstream/common/types.hpp"

namespace xpertstream {

/**
 * @brief Memory layout specification for static tensor shapes compiled for Intel NPU.
 * 
 * Intel NPU compiler requires fully static dimensions to optimize tiling, VPU scratchpad
 * memory allocation, and DMA descriptors without runtime graph re-compilations.
 */
struct StaticTensorShapeConfig {
    static constexpr batch_idx_t BATCH_SIZE = 1; // Locked to 1 for deterministic NPU scheduling
    seq_len_t max_seq_len{2048};                 // Locked context limit compiled into graph
    seq_len_t decode_step_len{1};                // Locked single-token decode shape [1, 1]

    [[nodiscard]] constexpr std::array<size_t, 2> decode_input_ids_shape() const noexcept {
        return {BATCH_SIZE, decode_step_len};
    }

    [[nodiscard]] constexpr std::array<size_t, 2> decode_position_ids_shape() const noexcept {
        return {BATCH_SIZE, decode_step_len};
    }

    [[nodiscard]] constexpr std::array<size_t, 2> attention_mask_shape() const noexcept {
        return {BATCH_SIZE, max_seq_len};
    }
};

} // namespace xpertstream
