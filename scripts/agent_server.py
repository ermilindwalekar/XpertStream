"""
XpertStream: Local OpenAI-Compatible Agent Server
=================================================
Serves an exported OpenVINO model (default: Qwen2.5-Coder-7B on NPU) behind a minimal
OpenAI-compatible `/v1/chat/completions` endpoint with tool-calling support, so the
XpertStream VS Code chat participant (or any OpenAI-compatible client) can drive it.

The NPU compiles static shapes, so the prompt budget (MAX_PROMPT_LEN) and the response
budget (MIN_RESPONSE_LEN) are fixed when the pipeline is created. Requests whose prompt
exceeds the budget are rejected with `context_length_exceeded` so the client can compact.
"""

import argparse
import json
import re
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 4 * 1024 * 1024
TOOL_OPEN = "<tool_call>"
TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)
FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def resolve_model_path(args) -> Path:
    models_root = Path("./models").resolve()
    if args.model_dir:
        return Path(args.model_dir).resolve()
    if args.model:
        for candidate in (Path(args.model), models_root / args.model, models_root / f"{args.model}-npu"):
            if (candidate / "openvino_model.xml").exists():
                return candidate.resolve()
    preferred = models_root / "qwen2.5-coder-7b-npu"
    if (preferred / "openvino_model.xml").exists():
        return preferred
    candidates = sorted(models_root.glob("*/openvino_model.xml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("[ERROR] No exported OpenVINO model found in ./models/. Run ExportNPUModel.cmd first.")
        sys.exit(1)
    return candidates[0].parent


def normalize_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def to_template_messages(messages):
    """Convert OpenAI chat messages to the shape expected by HF chat templates."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        if role == "developer":
            role = "system"
        msg = {"role": role, "content": normalize_content(m.get("content"))}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                arguments = fn.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        pass
                calls.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": arguments}})
            msg["tool_calls"] = calls
        out.append(msg)
    return out


def parse_tool_calls(text: str, tool_names: set):
    """Extract Qwen/Hermes-style <tool_call> blocks; fall back to fenced JSON naming a known tool."""
    matches = list(TOOL_BLOCK_RE.finditer(text))
    if not matches and tool_names:
        matches = list(FENCED_JSON_RE.finditer(text))
    raw_calls = [m.group(1) for m in matches]
    cut = matches[0].start() if matches else len(text)
    stripped = text.strip()
    if not raw_calls and tool_names and stripped.startswith("{") and stripped.endswith("}"):
        raw_calls, cut = [stripped], 0

    calls = []
    for raw in raw_calls:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") if isinstance(obj, dict) else None
        if not name or (tool_names and name not in tool_names):
            continue
        arguments = obj.get("arguments", obj.get("parameters", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        })
    content = text[:cut].strip() if calls else text.strip()
    return content, calls


def held_back_length(text: str, marker: str) -> int:
    """Length of the longest suffix of `text` that could be the start of `marker`."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-n:]):
            return n
    return 0


class ModelHost:
    def __init__(self, args):
        import openvino_genai as og
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging

        # Over-length prompts are rejected with context_length_exceeded, so the tokenizer warning is noise.
        hf_logging.set_verbosity_error()

        self.og = og
        self.model_path = resolve_model_path(args)
        self.model_name = self.model_path.name
        self.device = args.device
        self.max_prompt_len = args.max_prompt_len
        self.max_response_len = args.max_response_len
        self.lock = threading.Lock()
        self.last_error = None

        print(f"[*] Model:   {self.model_path}")
        print(f"[*] Device:  {self.device}  (prompt budget {self.max_prompt_len}, response budget {self.max_response_len})")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        t0 = time.perf_counter()
        if self.device == "NPU":
            cache_dir = self.model_path / "cache_npu"
            cache_dir.mkdir(exist_ok=True)
            self.pipe = og.LLMPipeline(
                str(self.model_path), "NPU",
                CACHE_DIR=str(cache_dir),
                MAX_PROMPT_LEN=self.max_prompt_len,
                MIN_RESPONSE_LEN=self.max_response_len,
            )
        else:
            self.pipe = og.LLMPipeline(str(self.model_path), self.device)
        print(f"[+] Pipeline ready in {time.perf_counter() - t0:.1f}s")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def build_prompt(self, messages, tools):
        return self.tokenizer.apply_chat_template(
            to_template_messages(messages),
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )

    def make_config(self, body, prompt_tokens):
        cfg = self.pipe.get_generation_config()
        cfg.apply_chat_template = False
        budget = self.max_prompt_len + self.max_response_len - prompt_tokens
        requested = body.get("max_tokens") or body.get("max_completion_tokens") or self.max_response_len
        cfg.max_new_tokens = max(1, min(int(requested), budget))
        temperature = float(body.get("temperature") or 0.0)
        if temperature > 0:
            cfg.do_sample = True
            cfg.temperature = temperature
            cfg.top_p = float(body.get("top_p") or 0.95)
        else:
            cfg.do_sample = False
        return cfg

    def generate(self, prompt, cfg, on_text):
        """Runs generation; `on_text` returns True to abort. Returns the full decoded text."""
        pieces = []
        running = getattr(self.og, "StreamingStatus", None)

        def streamer(subword):
            pieces.append(subword)
            stop = on_text(subword) if on_text else False
            if running is not None:
                return running.STOP if stop else running.RUNNING
            return bool(stop)

        with self.lock:
            try:
                self.pipe.generate(prompt, cfg, streamer)
            except OSError:
                raise
            except Exception as e:
                self.last_error = str(e).strip().splitlines()[-1] if str(e).strip() else repr(e)
                raise
        return "".join(pieces)


class ExclusiveHTTPServer(ThreadingHTTPServer):
    # Without this, Windows lets a second server silently bind the same port.
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    host: ModelHost = None
    server_version = "XpertStream/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")

    def send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error_json(self, status, message, code=None):
        self.send_json(status, {"error": {"message": message, "type": "invalid_request_error", "code": code}})

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self.send_json(200, {
                "status": "error" if self.host.last_error else "ok", "error": self.host.last_error,
                "model": self.host.model_name, "device": self.host.device,
                "max_prompt_len": self.host.max_prompt_len, "max_response_len": self.host.max_response_len,
            })
        elif self.path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": self.host.model_name, "object": "model", "owned_by": "xpertstream"}]})
        else:
            self.send_error_json(404, "Not found")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error_json(404, "Not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_error_json(413, "Request body missing or too large")
            return
        try:
            body = json.loads(self.rfile.read(length))
            messages = body["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a non-empty list")
        except (ValueError, KeyError) as e:
            self.send_error_json(400, f"Invalid request: {e}")
            return

        tools = body.get("tools") or []
        tool_names = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
        tool_names.discard(None)

        try:
            prompt = self.host.build_prompt(messages, tools)
        except Exception as e:
            self.send_error_json(400, f"Failed to apply chat template: {e}")
            return

        prompt_tokens = self.host.count_tokens(prompt)
        if prompt_tokens > self.host.max_prompt_len:
            no_tools = self.host.count_tokens(self.host.build_prompt(messages, None)) if tools else prompt_tokens
            system = sum(self.host.count_tokens(normalize_content(m.get("content")))
                         for m in messages if m.get("role") in ("system", "developer"))
            breakdown = (f"{len(tools)} tool definitions ~{prompt_tokens - no_tools}, system prompt ~{system}, "
                         f"other {len(messages)} messages ~{no_tools - system}")
            print(f"[reject] prompt={prompt_tokens} > budget={self.host.max_prompt_len}: {breakdown}")
            self.send_error_json(
                400,
                f"Prompt has {prompt_tokens} tokens but the server budget (--max-prompt-len) is "
                f"{self.host.max_prompt_len}. Breakdown: {breakdown}.",
                code="context_length_exceeded",
            )
            return

        cfg = self.host.make_config(body, prompt_tokens)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())
        t0 = time.perf_counter()

        if body.get("stream"):
            text = self.stream_completion(prompt, cfg, completion_id, created, tool_names)
        else:
            try:
                text = self.host.generate(prompt, cfg, None)
            except Exception as e:
                self.send_json(500, {"error": {"message": f"Generation failed: {self.host.last_error or e}",
                                               "type": "server_error", "code": "generation_failed"}})
                return
        if text is None:
            return  # client disconnected or error already reported

        content, calls = parse_tool_calls(text, tool_names)
        completion_tokens = self.host.count_tokens(text)
        finish = "tool_calls" if calls else ("length" if completion_tokens >= cfg.max_new_tokens else "stop")
        elapsed = time.perf_counter() - t0
        print(f"[gen] prompt={prompt_tokens} completion={completion_tokens} tools={len(calls)} "
              f"{elapsed:.1f}s ({completion_tokens / max(elapsed, 1e-6):.1f} tok/s)")

        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}

        if body.get("stream"):
            try:
                if calls:
                    self.write_chunk(completion_id, created, {"tool_calls": [
                        {"index": i, **c} for i, c in enumerate(calls)]})
                self.write_chunk(completion_id, created, {}, finish_reason=finish, usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except OSError:
                pass
            return

        message = {"role": "assistant", "content": content or None}
        if calls:
            message["tool_calls"] = calls
        self.send_json(200, {
            "id": completion_id, "object": "chat.completion", "created": created, "model": self.host.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        })

    def write_chunk(self, completion_id, created, delta, finish_reason=None, usage=None):
        chunk = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created, "model": self.host.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            chunk["usage"] = usage
        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def stream_completion(self, prompt, cfg, completion_id, created, tool_names):
        """Streams plain-text deltas; text from the first <tool_call> onward is withheld and parsed at the end."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        state = {"buf": "", "emitted": 0, "in_tool": False, "disconnected": False}

        def on_text(subword):
            state["buf"] += subword
            if state["in_tool"]:
                return False
            buf = state["buf"]
            idx = buf.find(TOOL_OPEN, state["emitted"])
            if idx >= 0:
                state["in_tool"] = True
                end = idx
            else:
                end = len(buf) - held_back_length(buf, TOOL_OPEN)
            if end > state["emitted"]:
                piece = buf[state["emitted"]:end]
                state["emitted"] = end
                try:
                    self.write_chunk(completion_id, created, {"role": "assistant", "content": piece})
                except OSError:
                    state["disconnected"] = True
                    return True
            return False

        try:
            self.write_chunk(completion_id, created, {"role": "assistant", "content": ""})
            text = self.host.generate(prompt, cfg, on_text)
        except OSError:
            return None
        except Exception as e:
            error = {"message": f"Generation failed: {self.host.last_error or e}", "type": "server_error",
                     "code": "generation_failed"}
            try:
                self.wfile.write(f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                pass
            return None
        if state["disconnected"]:
            return None
        if not state["in_tool"] and state["emitted"] < len(text):
            try:
                self.write_chunk(completion_id, created, {"content": text[state["emitted"]:]})
            except OSError:
                return None
        return text


def main():
    parser = argparse.ArgumentParser(description="XpertStream local OpenAI-compatible agent server")
    parser.add_argument("--device", "-d", choices=["NPU", "GPU", "CPU"], default="NPU")
    parser.add_argument("--model", type=str, default=None, help="Model folder name under ./models or a path")
    parser.add_argument("--model-dir", "-m", default=None, help="Path to exported OpenVINO model")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (keep on localhost)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-prompt-len", type=int, default=4096, help="Static NPU prompt budget (MAX_PROMPT_LEN)")
    parser.add_argument("--max-response-len", type=int, default=1024, help="Static NPU response budget (MIN_RESPONSE_LEN)")
    args = parser.parse_args()

    try:
        server = ExclusiveHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        print(f"[ERROR] Port {args.port} is already in use ({e}). Stop the other agent server or pass --port.")
        sys.exit(1)
    Handler.host = ModelHost(args)
    print(f"[+] XpertStream agent server listening on http://{args.host}:{args.port}/v1")
    print("    Open VS Code chat and talk to @xpert. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
