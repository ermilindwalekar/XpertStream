#include <iostream>
#include <vector>
#include <memory>
#include <span>
#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/adapter/llm_model_adapter.hpp"
#include "xpertstream/adapter/llama3_adapter.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"
#include "xpertstream/memory/stateful_kv_cache_tracker.hpp"
#include "xpertstream/runtime/sampler.hpp"
#include "xpertstream/runtime/greedy_sampler.hpp"
#include "xpertstream/agent/tool_sandbox.hpp"
#include "xpertstream/agent/react_state_machine.hpp"

using namespace xpertstream;

// Simple mock tool implementation for demonstration
class CalculatorTool : public ISandboxedTool {
public:
    [[nodiscard]] std::string name() const override { return "calculator"; }
    [[nodiscard]] std::string description() const override { return "Performs basic mathematical operations"; }
    [[nodiscard]] std::string input_schema_json() const override {
        return R"({"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]})";
    }
    [[nodiscard]] std::string output_schema_json() const override {
        return R"({"type":"object","properties":{"result":{"type":"number"}}})";
    }
    [[nodiscard]] std::string execute(const std::string& input_json) override {
        std::cout << "    [Tool Sandbox: Execute] Calculator received: " << input_json << "\n";
        return R"({"result": 42.0})";
    }
};

int main() {
    std::cout << "=========================================================\n";
    std::cout << "   XpertStream: Intel NPU Agentic Inference Engine Demo  \n";
    std::cout << "=========================================================\n\n";

    // 1. Static Shape Memory & Hard Sequence Limit Verification
    std::cout << "[Step 1] Configuring Static Tensor Shapes & KV-Cache Tracker...\n";
    StaticTensorShapeConfig shape_config{
        .max_seq_len = 16, // Intentionally small bound to demonstrate hard exception
        .decode_step_len = 1
    };
    StatefulKVCacheTracker tracker(shape_config, TargetNPUArch::NPU_40XX);
    std::cout << "  - Target Hardware: Intel Lunar Lake NPU (NPU_40XX)\n";
    std::cout << "  - Static Context Limit (max_seq_len): " << tracker.max_seq_len() << " tokens\n";
    std::cout << "  - Batch Size: " << StaticTensorShapeConfig::BATCH_SIZE << " (Locked)\n\n";

    // 2. Advancing KV-Cache progression within bounds
    std::cout << "[Step 2] Simulating prompt prefill (10 tokens)...\n";
    tracker.validate_sequence_boundary(10, "prefill");
    tracker.advance(10);
    std::cout << "  - Current Context: " << tracker.current_seq_len() << " tokens\n";
    std::cout << "  - Remaining Capacity: " << tracker.remaining_capacity() << " tokens\n\n";

    // 3. Testing Hard Exception Boundary (No CPU Fallback)
    std::cout << "[Step 3] Triggering Sequence Overflow Exception (Hard Boundary Test)...\n";
    try {
        std::cout << "  - Attempting to append 10 additional tokens (10 + 10 > 16)...\n";
        tracker.validate_sequence_boundary(10, "decode");
        tracker.advance(10);
    } catch (const SequenceOverflowException& ex) {
        std::cout << "  [SUCCESS] Caught hard boundary exception as expected:\n\n";
        std::cout << ex.what() << "\n\n";
    }

    // 4. Sandboxed Tool Registration
    std::cout << "[Step 4] Initializing Sandboxed Tool Registry...\n";
    auto tool_registry = std::make_shared<ToolRegistry>();
    tool_registry->register_tool(std::make_shared<CalculatorTool>());
    auto defs = tool_registry->get_tool_definitions();
    std::cout << "  - Registered tools: " << defs.size() << " (" << defs[0].function.name << ")\n\n";

    // 5. Decoupled ReAct State Machine Execution
    std::cout << "[Step 5] Running Decoupled ReAct State Machine Loop...\n";
    
    // Abstract inference invoker mock simulating a tool call followed by a final answer
    int turn = 0;
    InferenceInvoker mock_llm = [&turn](const std::vector<ChatMessage>& /*history*/, const std::vector<ToolDefinition>&) -> GenerationOutput {
        GenerationOutput out;
        if (turn == 0) {
            turn++;
            out.text = "Thinking: I need to calculate the value first.";
            ToolCall tc;
            tc.id = "call_abc123";
            tc.function.name = "calculator";
            tc.function.arguments_json = R"({"expression": "6 * 7"})";
            out.tool_calls.push_back(tc);
            out.finish_reason = FinishReason::ToolCalls;
        } else {
            out.text = "Based on the calculation, the answer is 42.";
            out.finish_reason = FinishReason::Stop;
        }
        return out;
    };

    ReActStateMachine::Config agent_cfg{
        .max_iterations = 5,
        .system_prompt = "You are a helpful assistant with calculator access."
    };
    ReActStateMachine agent(agent_cfg, tool_registry, mock_llm);

    std::string response = agent.run("What is 6 multiplied by 7?");
    std::cout << "\n[Agent Result]: " << response << "\n\n";

    // 6. Concrete LLM Model Adapter & Host Sampler Demonstration
    std::cout << "[Step 6] Demonstrating Llama-3 Model Adapter & Token Sampler...\n";
    Llama3Adapter adapter(2048);
    std::cout << "  - Adapter Family: " << adapter.metadata().model_family << "\n";
    std::cout << "  - Target Vocab Size: " << adapter.metadata().vocab_size << "\n";

    std::vector<ChatMessage> chat_test = {
        {Role::System, "You are an expert AI mathematician.", std::nullopt, std::nullopt, {}},
        {Role::User, "Calculate 15 + 27", std::nullopt, std::nullopt, {}}
    };
    std::string formatted_prompt = adapter.format_chat_text(chat_test, defs);
    std::cout << "  - Rendered Chat Prompt with Injected Tool Schemas:\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << formatted_prompt << "\n";
    std::cout << "---------------------------------------------------------\n";

    // Simulate model output parsing with tool calling syntax
    std::string model_raw_output = "I will compute this.\n<|python_tag|>calculator({\"expression\": \"15 + 27\"})";
    GenerationOutput parsed = adapter.parse_generation_output({}, model_raw_output);
    std::cout << "  - Parsed text: \"" << parsed.text << "\"\n";
    std::cout << "  - Extracted tool calls: " << parsed.tool_calls.size() << "\n";
    if (!parsed.tool_calls.empty()) {
        std::cout << "    * Tool: " << parsed.tool_calls[0].function.name << "\n";
        std::cout << "    * Args: " << parsed.tool_calls[0].function.arguments_json << "\n";
    }

    // Token sampler demonstration
    HostTokenSampler sampler;
    std::vector<float> sample_logits(10, 0.1f);
    sample_logits[7] = 8.5f; // high confidence for token 7
    SamplingParams sparams{.temperature = 0.0f, .top_p = 1.0f, .top_k = 1};
    token_id_t picked = sampler.sample(sample_logits, sparams, {});
    std::cout << "  - Sampler picked token index: " << picked << " (Expected: 7)\n";

    std::cout << "=========================================================\n";
    std::cout << "   XpertStream Architectural Demonstration Complete      \n";
    std::cout << "=========================================================\n";

    return 0;
}
