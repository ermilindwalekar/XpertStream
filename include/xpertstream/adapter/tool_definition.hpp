#pragma once

#include <string>

namespace xpertstream {

struct ToolFunctionDefinition {
    std::string name;
    std::string description;
    std::string parameters_schema_json; // JSON-Schema string
};

struct ToolDefinition {
    std::string type{"function"};
    ToolFunctionDefinition function;
};

} // namespace xpertstream
