#pragma once

#include <string>
#include <memory>
#include <span>
#include <filesystem>
#include "xpertstream/common/types.hpp"
#include "xpertstream/memory/static_tensor_desc.hpp"

namespace xpertstream {

struct NPUCompilerConfig {
    TargetNPUArch arch{TargetNPUArch::NPU_40XX};
    seq_len_t max_seq_len{2048};
    bool enable_turbo_mode{true};
    uint32_t dpu_groups{4}; // Hardware execution tiling groups
    std::string compilation_options{""};
};

/**
 * @brief Forward-declared handle wrapping low-level OpenVINO infer request.
 */
class INPUAsyncInferRequest {
public:
    virtual ~INPUAsyncInferRequest() = default;

    /**
     * @brief Set input tensor data pointer directly to avoid copies.
     */
    virtual void set_input_tensor(const std::string& tensor_name, std::span<const int64_t> data) = 0;
    virtual void set_input_tensor(const std::string& tensor_name, std::span<const token_id_t> data) = 0;

    /**
     * @brief Get output tensor memory span (e.g. logits).
     */
    [[nodiscard]] virtual std::span<const float> get_output_tensor(const std::string& tensor_name) = 0;

    /**
     * @brief Trigger non-blocking asynchronous dispatch to the Intel NPU command queue.
     */
    virtual void start_async() = 0;

    /**
     * @brief Block until current asynchronous NPU infer request finishes execution.
     */
    virtual void wait() = 0;

    /**
     * @brief Query if hardware compute is complete without blocking.
     */
    [[nodiscard]] virtual bool is_ready() const = 0;

    /**
     * @brief Reset stateful memory nodes (ReadValue/Assign) within this request.
     */
    virtual void reset_state() = 0;
};

/**
 * @brief Abstract representation of a model statically compiled for Intel NPU.
 */
class INPUCompiledModel {
public:
    virtual ~INPUCompiledModel() = default;

    /**
     * @brief Allocate a hardware infer request bound to this compiled model.
     */
    [[nodiscard]] virtual std::unique_ptr<INPUAsyncInferRequest> create_infer_request() = 0;

    [[nodiscard]] virtual const StaticTensorShapeConfig& shape_config() const noexcept = 0;
};

/**
 * @brief Hardware runtime abstraction barrier insulating the core engine from OpenVINO runtime.
 */
class INPUBackend {
public:
    virtual ~INPUBackend() = default;

    /**
     * @brief Compile an OpenVINO model graph with static dimensions strictly targeting NPU.
     * 
     * Applies OpenVINO NPU-specific properties:
     * - ov::hint::performance_mode = LATENCY
     * - ov::hint::execution_mode = ACCURACY/PERFORMANCE
     * - Static reshape: dynamic dimensions locked to [1, max_seq_len] and [1, 1]
     * - Strictly prevents CPU offload or fallback
     * 
     * @param model_path Path to the OpenVINO .xml IR file.
     * @param config Target static shape and hardware optimization parameters.
     * @return Statically compiled model handle.
     */
    [[nodiscard]] virtual std::shared_ptr<INPUCompiledModel> compile_static_model(
        const std::filesystem::path& model_path,
        const NPUCompilerConfig& config
    ) = 0;
};

} // namespace xpertstream
