"""
XpertStream: Local OpenAI-Compatible Agent Server
=================================================
Serves an exported OpenVINO model (default: Qwen2.5-Coder-7B on NPU) behind a local
OpenAI-compatible `/v1/chat/completions` endpoint. Agent frameworks use this API;
model profiles adapt each model family's tool-call format.

The NPU compiles static shapes, so the prompt budget (MAX_PROMPT_LEN) and the response
budget (MIN_RESPONSE_LEN) are fixed when the pipeline is created. Requests whose prompt
exceeds the budget are rejected with `context_length_exceeded` so the client can compact.
"""

import argparse
import json
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from configure_opencode import register_global_config
from inference_backend import create_pipeline
from model_profiles import available_profiles, select_profile

MAX_BODY_BYTES = 4 * 1024 * 1024


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
        self.log_output = args.log_output
        self.lock = threading.Lock()
        self.last_error = None

        print(f"[*] Model:   {self.model_path}")
        print(f"[*] Device:  {self.device}  (prompt budget {self.max_prompt_len}, response budget {self.max_response_len})")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.profile = select_profile(self.model_path, self.tokenizer.chat_template, args.profile)
        print(f"[*] Profile: {self.profile.name}  ({self.profile.description})")

        t0 = time.perf_counter()
        self.pipe = create_pipeline(og, self.model_path, self.device, self.max_prompt_len, self.max_response_len)
        print(f"[+] Pipeline ready in {time.perf_counter() - t0:.1f}s")
        self.check_tokenizer()

    def check_tokenizer(self):
        ov_tokenizer = self.pipe.get_tokenizer()
        try:
            warnings = self.profile.check_detokenizer(
                ov_tokenizer.decode, lambda s: ov_tokenizer.encode(s).input_ids.data.tolist()[0])
        except Exception as e:
            warnings = [f"Could not verify the detokenizer: {e}"]
        for w in warnings:
            print(f"[WARN] {w}")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def build_prompt(self, messages, tools):
        msgs = self.profile.prepare_messages(to_template_messages(messages))
        if tools:
            rules = self.profile.tool_rules()
            if msgs and msgs[0]["role"] == "system":
                msgs[0] = {**msgs[0], "content": f"{msgs[0]['content']}\n\n{rules}"}
            else:
                msgs.insert(0, {"role": "system", "content": rules})
        return self.tokenizer.apply_chat_template(
            msgs,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            **self.profile.chat_template_kwargs(),
        )

    def make_config(self, body, prompt_tokens):
        cfg = self.pipe.get_generation_config()
        cfg.apply_chat_template = False
        budget = self.max_prompt_len + self.max_response_len - prompt_tokens
        requested = body.get("max_tokens") or body.get("max_completion_tokens") or self.max_response_len
        cfg.max_new_tokens = max(1, min(int(requested), budget))
        defaults = self.profile.generation_defaults()
        temperature = float(body.get("temperature") if body.get("temperature") is not None
                             else defaults["temperature"])
        if temperature > 0:
            cfg.do_sample = True
            cfg.temperature = temperature
            cfg.top_p = float(body.get("top_p") if body.get("top_p") is not None else defaults["top_p"])
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

    def handle_error(self, request, client_address):
        # Clients that time out or cancel (e.g. health checks during model load) are not server errors.
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    host: ModelHost = None
    server_version = "XpertStream/0.1"

    def setup(self):
        super().setup()
        self.write_lock = threading.Lock()

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
                "model": self.host.model_name, "device": self.host.device, "profile": self.host.profile.name,
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
            text, emitted = self.stream_completion(prompt, cfg, completion_id, created, tool_names)
        else:
            try:
                text = self.host.generate(prompt, cfg, None)
            except Exception as e:
                self.send_json(500, {"error": {"message": f"Generation failed: {self.host.last_error or e}",
                                               "type": "server_error", "code": "generation_failed"}})
                return
        if text is None:
            return  # client disconnected or error already reported

        reply = self.host.profile.parse(text, tool_names)
        content, calls = reply.content, reply.tool_calls
        completion_tokens = self.host.count_tokens(text)
        finish = "tool_calls" if calls else ("length" if completion_tokens >= cfg.max_new_tokens else "stop")
        elapsed = time.perf_counter() - t0
        print(f"[gen] prompt={prompt_tokens} completion={completion_tokens} tools={len(calls)} "
              f"{elapsed:.1f}s ({completion_tokens / max(elapsed, 1e-6):.1f} tok/s)")
        if self.host.log_output:
            print(f"[out] offered {len(tool_names)} tools; raw model output:\n{text}\n[/out]")

        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                 "total_tokens": prompt_tokens + completion_tokens}

        if body.get("stream"):
            try:
                # Text withheld as a possible tool call that wasn't one still belongs to the reply.
                if not calls and emitted < len(text):
                    self.write_chunk(completion_id, created, {"content": text[emitted:]})
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
        self.write_raw(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))

    def write_raw(self, data: bytes):
        with self.write_lock:
            self.wfile.write(data)
            self.wfile.flush()

    def stream_completion(self, prompt, cfg, completion_id, created, tool_names):
        """Streams plain-text deltas; text from where the profile detects a tool call is withheld and parsed at the end."""
        profile = self.host.profile
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        state = {"buf": "", "emitted": 0, "in_tool": False, "disconnected": False}

        def on_text(subword):
            if state["disconnected"]:
                return True
            state["buf"] += subword
            if state["in_tool"]:
                return False
            buf = state["buf"]
            idx = profile.tool_call_start(buf)
            if idx is not None:
                state["in_tool"] = True
                end = idx
            else:
                end = len(buf) - profile.holdback(buf)
            if end > state["emitted"]:
                piece = buf[state["emitted"]:end]
                state["emitted"] = end
                try:
                    self.write_chunk(completion_id, created, {"role": "assistant", "content": piece})
                except OSError:
                    state["disconnected"] = True
                    return True
            return False

        # Long prompts can take minutes to prefill; SSE comments keep idle-timeout clients (Node fetch: 300s) alive.
        done = threading.Event()

        def heartbeat():
            while not done.wait(10):
                try:
                    self.write_raw(b": keepalive\n\n")
                except OSError:
                    state["disconnected"] = True
                    return

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            self.write_chunk(completion_id, created, {"role": "assistant", "content": ""})
            text = self.host.generate(prompt, cfg, on_text)
        except OSError:
            return None, 0
        except Exception as e:
            error = {"message": f"Generation failed: {self.host.last_error or e}", "type": "server_error",
                     "code": "generation_failed"}
            try:
                self.write_raw(f"data: {json.dumps({'error': error})}\n\ndata: [DONE]\n\n".encode("utf-8"))
            except OSError:
                pass
            return None, 0
        finally:
            done.set()
        if state["disconnected"]:
            return None, 0
        return text, state["emitted"]


def main():
    parser = argparse.ArgumentParser(description="XpertStream local OpenVINO model server")
    parser.add_argument("--device", "-d", choices=["NPU", "GPU", "GPU.0", "CPU"], default="NPU")
    parser.add_argument("--model", type=str, default=None, help="Model folder name under ./models or a path")
    parser.add_argument("--model-dir", "-m", default=None, help="Path to exported OpenVINO model")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (keep on localhost)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-prompt-len", type=int, default=4096, help="Static NPU prompt budget (MAX_PROMPT_LEN)")
    parser.add_argument("--max-response-len", type=int, default=1024, help="Static NPU response budget (MIN_RESPONSE_LEN)")
    parser.add_argument("--log-output", action="store_true", help="Print raw model output for each request (debugging)")
    parser.add_argument("--profile", choices=sorted(available_profiles()), default=None,
                        help="Model profile (tool-call format); auto-detected from the model by default")
    parser.add_argument("--register-opencode", action="store_true", help="Register the loaded model for OpenCode's VS Code extension")
    args = parser.parse_args()

    try:
        server = ExclusiveHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        print(f"[ERROR] Port {args.port} is already in use ({e}). Stop the other agent server or pass --port.")
        sys.exit(1)
    Handler.host = ModelHost(args)
    if args.register_opencode:
        register_global_config(
            {"status": "ok", "device": args.device, "max_prompt_len": args.max_prompt_len,
             "max_response_len": args.max_response_len},
            Handler.host.model_name, f"http://{args.host}:{args.port}")
    print(f"[+] XpertStream model server listening on http://{args.host}:{args.port}/v1")
    print("    Run OpenCode.cmd from the project you want to work on. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
