#pragma once

#include <memory>
#include <functional>
#include <future>
#include <span>
#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/memory/fixed_tensor_buffer.hpp"
#include "xpertstream/memory/stateful_kv_cache_tracker.hpp"
#include "xpertstream/runtime/npu_backend.hpp"
#include "xpertstream/runtime/sampler.hpp"

namespace xpertstream {

/**
 * @brief Token streaming callback interface.
 * Returns false if consumer requests generation abort.
 */
using TokenStreamCallback = std::function<bool(token_id_t token, const std::string& token_str)>;

/**
 * @brief Asynchronous Execution Pipeline decoupling NPU compute from host sampling.
 * 
 * Implements an overlapped execution loop:
 *  1. Asynchronously submit tensor evaluation pass (AsyncInferRequest::start_async).
 *  2. Concurrently on host CPU: execute sampling on prior step's logits, evaluate
 *     repetition penalties, evaluate tool call boundaries, and prepare input buffers.
 *  3. Synchronize hardware completion (AsyncInferRequest::wait), swap staging buffers,
 *     and immediately launch the subsequent NPU compute pass.
 */
class AsyncInferencePipeline {
public:
    AsyncInferencePipeline(
        std::shared_ptr<INPUCompiledModel> compiled_model,
        std::shared_ptr<ITokenSampler> sampler,
        StaticTensorShapeConfig shape_config,
        TargetNPUArch target_arch
    );
    ~AsyncInferencePipeline() = default;

    AsyncInferencePipeline(const AsyncInferencePipeline&) = delete;
    AsyncInferencePipeline& operator=(const AsyncInferencePipeline&) = delete;

    /**
     * @brief Prepare and populate stateful KV cache for the initial prompt sequence.
     * 
     * @param prompt_tokens Tokenized prompt array.
     * @throws SequenceOverflowException if prompt_tokens.size() > max_seq_len.
     */
    void prefill_prompt(std::span<const token_id_t> prompt_tokens);

    /**
     * @brief Execute asynchronous autoregressive decode loop.
     * 
     * Runs until a stop token is encountered, max generation steps reached, or
     * sequence capacity limit is reached (enforcing hard boundary without CPU fallback).
     * 
     * @param max_new_tokens Maximum generation steps.
     * @param sampling_params Sampling configuration.
     * @param stop_tokens List of token IDs that terminate generation.
     * @param on_token Optional streaming callback for each emitted token.
     * @return Sequence of generated token IDs.
     */
    [[nodiscard]] std::vector<token_id_t> generate_tokens(
        uint32_t max_new_tokens,
        const SamplingParams& sampling_params,
        std::span<const token_id_t> stop_tokens,
        TokenStreamCallback on_token = nullptr
    );

    /**
     * @brief Clear stateful KV-cache memory nodes on NPU and reset sequence counter.
     */
    void reset_context() noexcept;

    /**
     * @brief Read-only view of context tracker.
     */
    [[nodiscard]] const StatefulKVCacheTracker& tracker() const noexcept {
        return tracker_;
    }

private:
    std::shared_ptr<INPUCompiledModel> compiled_model_;
    std::shared_ptr<ITokenSampler> sampler_;
    std::unique_ptr<INPUAsyncInferRequest> infer_request_;
    FixedTensorBufferPool buffer_pool_;
    StatefulKVCacheTracker tracker_;
};

} // namespace xpertstream
