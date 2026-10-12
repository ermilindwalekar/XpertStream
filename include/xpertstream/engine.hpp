#pragma once

#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/adapter/llm_model_adapter.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"
#include "xpertstream/runtime/npu_backend.hpp"
#include "xpertstream/runtime/async_infer_pipeline.hpp"
#include "xpertstream/agent/tool_sandbox.hpp"
#include "xpertstream/agent/react_state_machine.hpp"

namespace xpertstream {

struct EngineConfig {
    std::filesystem::path model_ir_path;
    TargetNPUArch target_npu{TargetNPUArch::NPU_40XX};
    seq_len_t max_seq_len{2048};
    SamplingParams default_sampling;
    ReActStateMachine::Config agent_config;
};

/**
 * @brief Top-level Model-Agnostic Agentic Inference Engine.
 * 
 * Orchestrates:
 *  - Model adapters (OpenAI chat arrays -> tokens -> structured tool calls)
 *  - Static shape memory and ReadValue/Assign KV caches
 *  - Overlapped asynchronous NPU inference passes
 *  - Decoupled ReAct state machine and sandboxed tool registry
 */
class XpertStreamEngine {
public:
    XpertStreamEngine(
        EngineConfig config,
        std::shared_ptr<LLMModelAdapter> adapter,
        std::shared_ptr<INPUBackend> backend,
        std::shared_ptr<ITokenSampler> sampler,
        std::shared_ptr<ToolRegistry> tool_registry
    );

    ~XpertStreamEngine() = default;

    /**
     * @brief Initialize engine, compile static graph targeting Intel NPU, and allocate fixed buffers.
     */
    void initialize();

    /**
     * @brief Execute standard OpenAI-style chat completion with static shape bounds.
     * 
     * @param messages Vector of ChatMessage instances.
     * @param tools Optional tool definitions.
     * @param on_token Optional streaming callback.
     * @throws SequenceOverflowException if prompt or cumulative context exceeds max_seq_len.
     */
    [[nodiscard]] GenerationOutput chat_complete(
        const std::vector<ChatMessage>& messages,
        const std::vector<ToolDefinition>& tools = {},
        TokenStreamCallback on_token = nullptr
    );

    /**
     * @brief Run an autonomous agent loop resolving a user prompt using registered sandboxed tools.
     * 
     * @param prompt User task or question.
     * @return Final assistant response.
     */
    [[nodiscard]] std::string run_agent(const std::string& prompt);

    /**
     * @brief Direct access to tool registry to register custom sandboxed tools.
     */
    [[nodiscard]] std::shared_ptr<ToolRegistry> tool_registry() noexcept {
        return tool_registry_;
    }

    /**
     * @brief Access static context and memory tracker metrics.
     */
    [[nodiscard]] const StatefulKVCacheTracker& memory_tracker() const noexcept;

private:
    EngineConfig config_;
    std::shared_ptr<LLMModelAdapter> adapter_;
    std::shared_ptr<INPUBackend> backend_;
    std::shared_ptr<ITokenSampler> sampler_;
    std::shared_ptr<ToolRegistry> tool_registry_;
    std::shared_ptr<INPUCompiledModel> compiled_model_;
    std::unique_ptr<AsyncInferencePipeline> pipeline_;
};

} // namespace xpertstream
