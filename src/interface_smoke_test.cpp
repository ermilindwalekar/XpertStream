#include "xpertstream/common/types.hpp"
#include "xpertstream/common/exceptions.hpp"
#include "xpertstream/adapter/llm_model_adapter.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"
#include "xpertstream/memory/fixed_tensor_buffer.hpp"
#include "xpertstream/memory/stateful_kv_cache_tracker.hpp"
#include "xpertstream/runtime/sampler.hpp"
#include "xpertstream/runtime/npu_backend.hpp"
#include "xpertstream/runtime/async_infer_pipeline.hpp"
#include "xpertstream/agent/tool_sandbox.hpp"
#include "xpertstream/agent/react_state_machine.hpp"
#include "xpertstream/runtime/greedy_sampler.hpp"
#include "xpertstream/runtime/openvino_npu_backend.hpp"
#include "xpertstream/adapter/llama3_adapter.hpp"
#include "xpertstream/engine.hpp"

// Compile-time static assertions validating type traits and signatures
static_assert(std::is_abstract_v<xpertstream::LLMModelAdapter>, "LLMModelAdapter must be an abstract interface");
static_assert(std::is_abstract_v<xpertstream::INPUBackend>, "INPUBackend must be an abstract interface");
static_assert(std::is_abstract_v<xpertstream::INPUAsyncInferRequest>, "INPUAsyncInferRequest must be an abstract interface");
static_assert(std::is_abstract_v<xpertstream::ISandboxedTool>, "ISandboxedTool must be an abstract interface");
static_assert(std::is_base_of_v<xpertstream::EngineException, xpertstream::SequenceOverflowException>, "Must derive from EngineException");
