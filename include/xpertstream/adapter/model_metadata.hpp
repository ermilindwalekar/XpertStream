#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "xpertstream/common/types.hpp"

namespace xpertstream {

struct SpecialTokens {
    token_id_t bos_token_id{-1};
    token_id_t eos_token_id{-1};
    token_id_t pad_token_id{-1};
    std::vector<token_id_t> stop_token_ids;
};

struct ModelMetadata {
    std::string model_name;
    std::string model_family; // e.g. "Llama-3", "Qwen-2.5", "Mistral", "Phi-3.5"
    uint32_t vocab_size{0};
    seq_len_t max_seq_len{2048};
    uint32_t num_layers{0};
    uint32_t num_attention_heads{0};
    uint32_t num_key_value_heads{0};
    uint32_t hidden_size{0};
    SpecialTokens special_tokens;
};

} // namespace xpertstream
