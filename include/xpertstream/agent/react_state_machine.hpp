#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "xpertstream/common/types.hpp"
#include "xpertstream/adapter/tool_definition.hpp"
#include "xpertstream/agent/tool_sandbox.hpp"

namespace xpertstream {

enum class ReActState {
    Idle,
    Reasoning,            // Awaiting model generation (Thought / Action plan)
    ActionDispatch,       // Verifying tool call arguments against schema
    ObservationPending,   // Executing sandboxed tool
    Completed,            // Final user-facing response generated
    Failed                // Hard sequence overflow, validation failure, or max steps reached
};

struct ReActStepResult {
    ReActState current_state{ReActState::Idle};
    std::string thought_or_response;
    std::vector<ToolCall> dispatched_actions;
    std::vector<std::string> observations;
    bool is_terminal{false};
};

/**
 * @brief Abstract callable delegate decoupled from hardware execution.
 * 
 * Invokes model inference without coupling the agent state machine to OpenVINO,
 * tensors, or NPU runtimes.
 */
using InferenceInvoker = std::function<GenerationOutput(
    const std::vector<ChatMessage>& chat_history,
    const std::vector<ToolDefinition>& active_tools
)>;

/**
 * @brief Code-first ReAct (Reasoning-Action-Observation) State Machine.
 * 
 * Manages deterministic turn-based agent execution loops completely independent
 * of the underlying hardware inference runtime.
 */
class ReActStateMachine {
public:
    struct Config {
        uint32_t max_iterations{10};
        std::string system_prompt{"You are an autonomous AI agent capable of using tools to resolve user queries."};
    };

    ReActStateMachine(
        Config config,
        std::shared_ptr<ToolRegistry> tool_registry,
        InferenceInvoker inference_invoker
    );

    ~ReActStateMachine() = default;

    /**
     * @brief Initialize state machine with user input prompt.
     */
    void start(const std::string& user_query);

    /**
     * @brief Execute a single transition of the ReAct state machine.
     * 
     * @return Result of the step, including state and emitted actions/observations.
     */
    ReActStepResult step();

    /**
     * @brief Run the state machine loop until Completed or Failed.
     * 
     * @return Final assistant message text.
     */
    std::string run(const std::string& user_query);

    [[nodiscard]] ReActState current_state() const noexcept { return state_; }
    [[nodiscard]] const std::vector<ChatMessage>& conversation_history() const noexcept { return history_; }
    [[nodiscard]] uint32_t current_iteration() const noexcept { return iteration_; }

private:
    Config config_;
    std::shared_ptr<ToolRegistry> tool_registry_;
    InferenceInvoker inference_invoker_;
    ReActState state_{ReActState::Idle};
    uint32_t iteration_{0};
    std::vector<ChatMessage> history_;
    std::vector<ToolCall> pending_actions_;
};

} // namespace xpertstream
