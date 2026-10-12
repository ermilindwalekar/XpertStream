#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include "xpertstream/adapter/tool_definition.hpp"
#include "xpertstream/common/exceptions.hpp"

namespace xpertstream {

/**
 * @brief Thrown when tool execution inputs or outputs violate registered JSON-Schema.
 */
class ToolSchemaValidationException final : public EngineException {
public:
    explicit ToolSchemaValidationException(std::string message)
        : EngineException(std::move(message)) {}
};

/**
 * @brief Interface for sandboxed tools with JSON-Schema verification contracts.
 */
class ISandboxedTool {
public:
    virtual ~ISandboxedTool() = default;

    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual std::string input_schema_json() const = 0;
    [[nodiscard]] virtual std::string output_schema_json() const = 0;

    /**
     * @brief Execute the tool logic with verified JSON input string.
     * 
     * @param input_json Validated input JSON payload conforming to input_schema_json().
     * @return Output string conforming to output_schema_json().
     */
    [[nodiscard]] virtual std::string execute(const std::string& input_json) = 0;
};

/**
 * @brief Tool registry enforcing input/output schema validation and sandbox isolation.
 */
class ToolRegistry {
public:
    ToolRegistry() = default;
    ~ToolRegistry() = default;

    /**
     * @brief Register a new sandboxed tool.
     */
    void register_tool(std::shared_ptr<ISandboxedTool> tool);

    /**
     * @brief Retrieve tool definitions for injection into chat prompt adapters.
     */
    [[nodiscard]] std::vector<ToolDefinition> get_tool_definitions() const;

    /**
     * @brief Invoke a tool by name with JSON Schema pre-validation and post-validation.
     * 
     * @param tool_name Registered function identifier.
     * @param arguments_json Raw JSON argument string from model output.
     * @return Observation string from tool execution.
     * @throws ToolSchemaValidationException if arguments do not conform to schema.
     */
    [[nodiscard]] std::string invoke(
        const std::string& tool_name,
        const std::string& arguments_json
    );

    [[nodiscard]] bool has_tool(const std::string& tool_name) const noexcept;

private:
    std::unordered_map<std::string, std::shared_ptr<ISandboxedTool>> tools_;

    void validate_json_schema(const std::string& json_str, const std::string& schema_str, const std::string& context_tag);
};

} // namespace xpertstream
