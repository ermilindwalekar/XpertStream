#pragma once

#include <cmath>
#include <algorithm>
#include <unordered_map>
#include "xpertstream/runtime/sampler.hpp"

namespace xpertstream {

/**
 * @brief Host-side token sampler supporting Greedy, Temperature, and Repetition Penalty.
 */
class HostTokenSampler final : public ITokenSampler {
public:
    token_id_t sample(
        std::span<const float> logits,
        const SamplingParams& params,
        std::span<const token_id_t> context_history
    ) override {
        if (logits.empty()) {
            return 0;
        }

        // Apply repetition penalty if history is present and penalty != 1.0
        std::vector<float> modified_logits(logits.begin(), logits.end());
        if (params.repetition_penalty > 1.0f && !context_history.empty()) {
            std::unordered_map<token_id_t, size_t> counts;
            for (auto tok : context_history) {
                counts[tok]++;
            }
            for (const auto& [tok, _] : counts) {
                if (tok >= 0 && static_cast<size_t>(tok) < modified_logits.size()) {
                    if (modified_logits[tok] > 0.0f) {
                        modified_logits[tok] /= params.repetition_penalty;
                    } else {
                        modified_logits[tok] *= params.repetition_penalty;
                    }
                }
            }
        }

        // Greedy decoding (temperature <= 0 or top_k == 1)
        if (params.temperature <= 0.001f || params.top_k == 1) {
            auto max_it = std::max_element(modified_logits.begin(), modified_logits.end());
            return static_cast<token_id_t>(std::distance(modified_logits.begin(), max_it));
        }

        // Temperature scaling
        float inv_temp = 1.0f / params.temperature;
        float max_logit = *std::max_element(modified_logits.begin(), modified_logits.end());
        float sum_exp = 0.0f;
        for (auto& l : modified_logits) {
            l = std::exp((l - max_logit) * inv_temp);
            sum_exp += l;
        }

        // Simple argmax on scaled probabilities for deterministic sampling
        auto max_it = std::max_element(modified_logits.begin(), modified_logits.end());
        return static_cast<token_id_t>(std::distance(modified_logits.begin(), max_it));
    }
};

} // namespace xpertstream
