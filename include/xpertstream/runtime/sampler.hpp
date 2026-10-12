#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include "xpertstream/common/types.hpp"

namespace xpertstream {

struct SamplingParams {
    float temperature{0.0f}; // 0.0 = greedy
    float top_p{1.0f};
    uint32_t top_k{1};
    float repetition_penalty{1.0f};
};

/**
 * @brief Host-side token sampler decoupled from NPU tensor compute pass.
 */
class ITokenSampler {
public:
    virtual ~ITokenSampler() = default;

    /**
     * @brief Sample next token from output logits while NPU begins next pass or remains idle.
     * 
     * @param logits Output logits tensor slice over vocabulary [vocab_size].
     * @param params Sampling configuration (greedy, temperature, etc.).
     * @param context_history Previously emitted tokens for repetition penalty.
     * @return Selected token_id_t.
     */
    [[nodiscard]] virtual token_id_t sample(
        std::span<const float> logits,
        const SamplingParams& params,
        std::span<const token_id_t> context_history
    ) = 0;
};

} // namespace xpertstream
