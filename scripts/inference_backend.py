"""OpenVINO pipeline construction, independent of the agent and HTTP transport."""

from pathlib import Path


def create_pipeline(genai, model_path: Path, device: str, max_prompt_len: int, max_response_len: int):
    if device == "NPU":
        cache_dir = model_path / "cache_npu"
        cache_dir.mkdir(exist_ok=True)
        return genai.LLMPipeline(
            str(model_path), "NPU",
            CACHE_DIR=str(cache_dir),
            MAX_PROMPT_LEN=max_prompt_len,
            MIN_RESPONSE_LEN=max_response_len,
        )
    return genai.LLMPipeline(str(model_path), device)