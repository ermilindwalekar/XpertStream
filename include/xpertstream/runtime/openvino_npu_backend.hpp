#pragma once

#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include "xpertstream/runtime/npu_backend.hpp"
#include "xpertstream/common/exceptions.hpp"

#if __has_include(<openvino/openvino.hpp>)
#define XPERTSTREAM_OPENVINO_AVAILABLE 1
#include <openvino/openvino.hpp>
#endif

namespace xpertstream {

#if XPERTSTREAM_OPENVINO_AVAILABLE

class OpenVINOAsyncInferRequest final : public INPUAsyncInferRequest {
public:
    explicit OpenVINOAsyncInferRequest(ov::InferRequest infer_request)
        : infer_request_(std::move(infer_request)) {}

    void set_input_tensor(const std::string& tensor_name, std::span<const int64_t> data) override {
        ov::Shape shape = {1, data.size()};
        ov::Tensor tensor(ov::element::i64, shape, const_cast<int64_t*>(data.data()));
        infer_request_.set_tensor(tensor_name, tensor);
    }

    void set_input_tensor(const std::string& tensor_name, std::span<const token_id_t> data) override {
        ov::Shape shape = {1, data.size()};
        ov::Tensor tensor(ov::element::i32, shape, const_cast<token_id_t*>(data.data()));
        infer_request_.set_tensor(tensor_name, tensor);
    }

    [[nodiscard]] std::span<const float> get_output_tensor(const std::string& tensor_name) override {
        ov::Tensor out_tensor = infer_request_.get_tensor(tensor_name);
        const float* ptr = out_tensor.data<const float>();
        size_t count = out_tensor.get_size();
        return std::span<const float>(ptr, count);
    }

    void start_async() override {
        infer_request_.start_async();
    }

    void wait() override {
        infer_request_.wait();
    }

    [[nodiscard]] bool is_ready() const override {
        return infer_request_.wait_for(std::chrono::milliseconds(0));
    }

    void reset_state() override {
        infer_request_.reset_state();
    }

private:
    ov::InferRequest infer_request_;
};

class OpenVINOCompiledModel final : public INPUCompiledModel {
public:
    OpenVINOCompiledModel(ov::CompiledModel compiled_model, StaticTensorShapeConfig shape_config)
        : compiled_model_(std::move(compiled_model)), shape_config_(shape_config) {}

    [[nodiscard]] std::unique_ptr<INPUAsyncInferRequest> create_infer_request() override {
        return std::make_unique<OpenVINOAsyncInferRequest>(compiled_model_.create_infer_request());
    }

    [[nodiscard]] const StaticTensorShapeConfig& shape_config() const noexcept override {
        return shape_config_;
    }

private:
    ov::CompiledModel compiled_model_;
    StaticTensorShapeConfig shape_config_;
};

class OpenVINONPUBackend final : public INPUBackend {
public:
    OpenVINONPUBackend() {
        // Query available devices
        auto devices = core_.get_available_devices();
        bool has_npu = false;
        for (const auto& dev : devices) {
            if (dev.find("NPU") != std::string::npos) {
                has_npu = true;
                break;
            }
        }
        target_device_ = has_npu ? "NPU" : "CPU"; // Fallback only for developer workstations lacking physical NPU
    }

    [[nodiscard]] std::shared_ptr<INPUCompiledModel> compile_static_model(
        const std::filesystem::path& model_path,
        const NPUCompilerConfig& config
    ) override {
        if (!std::filesystem::exists(model_path)) {
            throw EngineException("Model IR path does not exist: " + model_path.string());
        }

        // 1. Read OpenVINO IR model
        auto model = core_.read_model(model_path.string());

        // 2. Lock static dimensions: batch=1, decode_len=1, max_seq_len
        // Reshape dynamic inputs to static shapes required by NPU compiler
        std::map<ov::Output<ov::Node>, ov::PartialShape> reshape_map;
        for (const auto& input : model->inputs()) {
            const auto& name = input.get_any_name();
            if (name.find("input_ids") != std::string::npos || name.find("position_ids") != std::string::npos) {
                reshape_map[input] = ov::PartialShape{1, 1};
            } else if (name.find("attention_mask") != std::string::npos) {
                reshape_map[input] = ov::PartialShape{1, static_cast<int64_t>(config.max_seq_len)};
            } else if (name.find("beam_idx") != std::string::npos) {
                reshape_map[input] = ov::PartialShape{1};
            }
        }
        if (!reshape_map.empty()) {
            model->reshape(reshape_map);
        }

        // 3. Configure hardware properties for Intel NPU
        ov::AnyMap npu_properties = {
            {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
            {ov::hint::execution_mode.name(), ov::hint::ExecutionMode::PERFORMANCE}
        };

        // 4. Compile strictly for NPU
        auto compiled = core_.compile_model(model, target_device_, npu_properties);

        StaticTensorShapeConfig shape_cfg{
            .max_seq_len = config.max_seq_len,
            .decode_step_len = 1
        };

        return std::make_shared<OpenVINOCompiledModel>(std::move(compiled), shape_cfg);
    }

private:
    ov::Core core_;
    std::string target_device_{"NPU"};
};

#else

/**
 * @brief Reference stub for environments where OpenVINO headers are not yet installed.
 */
class OpenVINONPUBackend final : public INPUBackend {
public:
    [[nodiscard]] std::shared_ptr<INPUCompiledModel> compile_static_model(
        const std::filesystem::path& model_path,
        const NPUCompilerConfig& /*config*/
    ) override {
        throw EngineException(
            "OpenVINO runtime headers not detected during build. "
            "To enable live hardware execution, install Intel OpenVINO C++ runtime (2024.4+ / 2025.x) "
            "and build with -DXPERTSTREAM_ENABLE_OPENVINO=ON. Attempted to load: " + model_path.string()
        );
    }
};

#endif

} // namespace xpertstream
