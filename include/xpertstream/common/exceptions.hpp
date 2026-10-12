#pragma once

#include <exception>
#include <string>
#include <sstream>
#include <cstdint>
#include "xpertstream/common/types.hpp"

namespace xpertstream {

/**
 * @brief Base runtime exception for XpertStream low-level engine failures.
 */
class EngineException : public std::exception {
public:
    explicit EngineException(std::string message) : message_(std::move(message)) {}
    [[nodiscard]] const char* what() const noexcept override {
        return message_.c_str();
    }
protected:
    std::string message_;
};

/**
 * @brief Hard boundary exception thrown when sequence length exceeds statically allocated NPU graph bounds.
 * 
 * In static-shape compilation modes, dynamic graph recompilations and CPU fallback/offloading
 * are strictly prohibited to maintain deterministic real-time NPU throughput and memory bounds.
 */
class SequenceOverflowException final : public EngineException {
public:
    struct DiagnosticMetrics {
        seq_len_t current_seq_len{0};
        seq_len_t max_seq_len{0};
        seq_len_t requested_additional_tokens{0};
        TargetNPUArch target_arch{TargetNPUArch::NPU_40XX};
        const char* execution_phase{"unknown"};
    };

    explicit SequenceOverflowException(DiagnosticMetrics metrics)
        : EngineException(format_diagnostic_message(metrics)),
          metrics_(metrics) {}

    [[nodiscard]] const DiagnosticMetrics& diagnostics() const noexcept {
        return metrics_;
    }

private:
    DiagnosticMetrics metrics_;

    static std::string format_diagnostic_message(const DiagnosticMetrics& m) {
        std::ostringstream oss;
        oss << "[XpertStream::FATAL] SequenceOverflowException: Cumulative context length violation.\n"
            << "  - Phase:              " << m.execution_phase << "\n"
            << "  - Current Length:     " << m.current_seq_len << " tokens\n"
            << "  - Requested Addition: " << m.requested_additional_tokens << " tokens\n"
            << "  - Static Allocation:  " << m.max_seq_len << " tokens (Hard Bound)\n"
            << "  - Target Hardware:    NPU Architecture (Index " << static_cast<int>(m.target_arch) << ")\n"
            << "  - Policy:             CPU Offloading and Dynamic Graph Fallback are STRICTLY DISABLED.\n"
            << "  - Diagnostic:         Total sequence (" << (m.current_seq_len + m.requested_additional_tokens)
            << ") exceeds pre-compiled static tensor buffer limit (" << m.max_seq_len << ").";
        return oss.str();
    }
};

/**
 * @brief Thrown when input tensors or batch dimensions deviate from static compilation shape.
 */
class StaticShapeViolationException final : public EngineException {
public:
    explicit StaticShapeViolationException(std::string message)
        : EngineException(std::move(message)) {}
};

/**
 * @brief Thrown when NPU driver, firmware, or hardware queue encounters unrecoverable state.
 */
class NPUHardwareException final : public EngineException {
public:
    explicit NPUHardwareException(std::string message, int32_t error_code = -1)
        : EngineException(std::move(message)), error_code_(error_code) {}

    [[nodiscard]] int32_t error_code() const noexcept { return error_code_; }

private:
    int32_t error_code_;
};

} // namespace xpertstream
