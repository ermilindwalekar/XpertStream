import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from inference_backend import create_pipeline


class FakeGenAI:
    def LLMPipeline(self, *args, **kwargs):
        return args, kwargs


class PipelineTests(unittest.TestCase):
    def test_npu_uses_static_budgets_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            args, kwargs = create_pipeline(FakeGenAI(), model, "NPU", 4096, 512)
            self.assertEqual(args, (str(model), "NPU"))
            self.assertEqual(kwargs, {
                "CACHE_DIR": str(model / "cache_npu"),
                "MAX_PROMPT_LEN": 4096,
                "MIN_RESPONSE_LEN": 512,
            })
            self.assertTrue((model / "cache_npu").is_dir())

    def test_intel_igpu_preserves_device_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            args, kwargs = create_pipeline(FakeGenAI(), model, "GPU.0", 8192, 1024)
            self.assertEqual(args, (str(model), "GPU.0"))
            self.assertEqual(kwargs, {})
            self.assertFalse((model / "cache_npu").exists())

    def test_cpu_keeps_the_existing_non_npu_path(self):
        args, kwargs = create_pipeline(FakeGenAI(), Path("model"), "CPU", 2048, 256)
        self.assertEqual(args, ("model", "CPU"))
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()