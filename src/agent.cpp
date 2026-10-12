#include <iostream>
#include <stdexcept>
#include "xpertstream/agent/tool_sandbox.hpp"
#include "xpertstream/agent/react_state_machine.hpp"

namespace xpertstream {

void ToolRegistry::register_tool(std::shared_ptr<ISandboxedTool> tool) {
    if (tool) {
        tools_[tool->name()] = tool;
    }
}

std::vector<ToolDefinition> ToolRegistry::get_tool_definitions() const {
    std::vector<ToolDefinition> defs;
    defs.reserve(tools_.size());
    for (const auto& [name, tool] : tools_) {
        ToolDefinition def;
        def.type = "function";
        def.function.name = tool->name();
        def.function.description = tool->description();
        def.function.parameters_schema_json = tool->input_schema_json();
        defs.push_back(def);
    }
    return defs;
}

std::string ToolRegistry::invoke(const std::string& tool_name, const std::string& arguments_json) {
    auto it = tools_.find(tool_name);
    if (it == tools_.end()) {
        throw ToolSchemaValidationException("Tool '" + tool_name + "' is not registered.");
    }
    validate_json_schema(arguments_json, it->second->input_schema_json(), "input");
    std::string result = it->second->execute(arguments_json);
    validate_json_schema(result, it->second->output_schema_json(), "output");
    return result;
}

bool ToolRegistry::has_tool(const std::string& tool_name) const noexcept {
    return tools_.find(tool_name) != tools_.end();
}

void ToolRegistry::validate_json_schema(const std::string& json_str, const std::string& /*schema_str*/, const std::string& /*context_tag*/) {
    // In production, invoke a compiled JSON-Schema validator (such as valijson or nlohmann::json-schema-validator)
    if (json_str.empty()) {
        throw ToolSchemaValidationException("Schema validation failed: empty JSON payload.");
    }
}

ReActStateMachine::ReActStateMachine(
    Config config,
    std::shared_ptr<ToolRegistry> tool_registry,
    InferenceInvoker inference_invoker
) : config_(std::move(config)),
    tool_registry_(std::move(tool_registry)),
    inference_invoker_(std::move(inference_invoker)) {}

void ReActStateMachine::start(const std::string& user_query) {
    history_.clear();
    pending_actions_.clear();
    iteration_ = 0;
    state_ = ReActState::Reasoning;

    if (!config_.system_prompt.empty()) {
        ChatMessage sys_msg;
        sys_msg.role = Role::System;
        sys_msg.content = config_.system_prompt;
        history_.push_back(sys_msg);
    }

    ChatMessage user_msg;
    user_msg.role = Role::User;
    user_msg.content = user_query;
    history_.push_back(user_msg);
}

ReActStepResult ReActStateMachine::step() {
    ReActStepResult result;

    if (iteration_ >= config_.max_iterations) {
        state_ = ReActState::Failed;
        result.current_state = state_;
        result.thought_or_response = "Maximum reasoning iterations exceeded.";
        result.is_terminal = true;
        return result;
    }

    iteration_++;

    if (state_ == ReActState::Reasoning) {
        auto tool_defs = tool_registry_ ? tool_registry_->get_tool_definitions() : std::vector<ToolDefinition>{};
        GenerationOutput output = inference_invoker_(history_, tool_defs);

        if (!output.tool_calls.empty()) {
            pending_actions_ = output.tool_calls;
            result.dispatched_actions = output.tool_calls;
            result.thought_or_response = output.text;

            ChatMessage assistant_msg;
            assistant_msg.role = Role::Assistant;
            assistant_msg.content = output.text;
            assistant_msg.tool_calls = output.tool_calls;
            history_.push_back(assistant_msg);

            state_ = ReActState::ActionDispatch;
            result.current_state = state_;
            return result;
        } else {
            result.thought_or_response = output.text;
            state_ = ReActState::Completed;
            result.current_state = state_;
            result.is_terminal = true;
            return result;
        }
    } else if (state_ == ReActState::ActionDispatch) {
        for (const auto& action : pending_actions_) {
            std::string observation;
            try {
                observation = tool_registry_->invoke(action.function.name, action.function.arguments_json);
            } catch (const std::exception& ex) {
                observation = std::string("Error: ") + ex.what();
            }

            ChatMessage tool_msg;
            tool_msg.role = Role::Tool;
            tool_msg.name = action.function.name;
            tool_msg.tool_call_id = action.id;
            tool_msg.content = observation;
            history_.push_back(tool_msg);

            result.observations.push_back(observation);
        }
        pending_actions_.clear();
        state_ = ReActState::Reasoning;
        result.current_state = state_;
        return result;
    }

    result.current_state = state_;
    result.is_terminal = (state_ == ReActState::Completed || state_ == ReActState::Failed);
    return result;
}

std::string ReActStateMachine::run(const std::string& user_query) {
    start(user_query);
    while (state_ != ReActState::Completed && state_ != ReActState::Failed) {
        auto res = step();
        if (res.is_terminal && state_ == ReActState::Completed) {
            return res.thought_or_response;
        }
    }
    return (state_ == ReActState::Completed && !history_.empty()) ? history_.back().content : "Agent failed to complete.";
}

} // namespace xpertstream
