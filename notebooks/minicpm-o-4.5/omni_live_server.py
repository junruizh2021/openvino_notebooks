#!/usr/bin/env python3
"""Launch a local web demo for MiniCPM-o 4.5 OpenVINO inference.

The demo exposes text, image, and audio inputs through Gradio and loads the
converted OpenVINO model from a local directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from collections import deque
from pathlib import Path


DEFAULT_MODEL_PATH = Path("/home/nvme-data/AI-models/MiniCPM-o-4_5-OV")
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7860

REQUIRED_MODEL_FILES = (
    "config.json",
    "openvino_llm_embedding_model.xml",
    "openvino_llm_language_model.xml",
    "openvino_vision_model.xml",
    "openvino_resampler_model.xml",
    "openvino_audio_encoder_model.xml",
    "openvino_audio_projection_model.xml",
    "openvino_tts_embedding_model.xml",
    "openvino_tts_language_model.xml",
    "openvino_hift_model.xml",
)


def ensure_localhost_proxy_bypass() -> None:
    """Keep Gradio startup-event requests off corporate/system proxies."""
    required = ("localhost", "127.0.0.1", "0.0.0.0")
    for key in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(key, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        for host in required:
            if host not in entries:
                entries.append(host)
        os.environ[key] = ",".join(entries)


def configure_matplotlib_for_gradio() -> None:
    """Initialize Matplotlib in the main thread before Gradio request handling."""
    base_tmp = Path(tempfile.gettempdir()) / f"minicpmo_{os.getuid()}"
    base_tmp.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("GRADIO_TEMP_DIR", str(base_tmp / "gradio"))
    Path(os.environ["GRADIO_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

    mpl_dir = base_tmp / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        matplotlib.get_backend()
    except Exception as exc:
        print(f"⚠️ Matplotlib pre-initialization failed: {exc}")


def ensure_ffmpeg_path() -> None:
    """Make user-local ffmpeg/ffprobe visible to Gradio streaming audio."""
    user_bin = Path.home() / "bin"
    if user_bin.exists():
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        if str(user_bin) not in path_entries:
            os.environ["PATH"] = str(user_bin) + os.pathsep + current_path

    if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
        print("⚠️ ffmpeg/ffprobe not found. Gradio Audio(streaming=True) may fail.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MiniCPM-o 4.5 OpenVINO multimodal web demo.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the converted OpenVINO model directory. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("OV_DEVICE", "CPU"),
        help="OpenVINO device for LLM, vision, and audio encoders. Default: CPU",
    )
    parser.add_argument(
        "--tts-device",
        default=os.getenv("OV_TTS_DEVICE"),
        help="OpenVINO device for TTS submodels. Default: same as --device",
    )
    parser.add_argument(
        "--tts-flow-device",
        default=os.getenv("OV_TTS_FLOW_DEVICE"),
        help="OpenVINO device for Flow embedding/encoder helper submodels. Default: same as --tts-device",
    )
    parser.add_argument(
        "--ov-performance-hint",
        default=os.getenv("OV_PERFORMANCE_HINT", "LATENCY"),
        help="OpenVINO PERFORMANCE_HINT for compiled models. Use LATENCY for realtime TTS. Default: LATENCY",
    )
    parser.add_argument(
        "--ov-cache-dir",
        type=Path,
        default=None,
        help="OpenVINO compiled model cache directory. Default: ~/.cache/openvino/minicpm-o-4.5/<model-dir-name>",
    )
    parser.add_argument(
        "--mode",
        choices=("chat", "tts", "ws-tts", "omni"),
        default="chat",
        help="Demo mode: full multimodal chat, Gradio TTS page, WebSocket low-latency TTS, or omni-live (video+audio→speech+text). Default: chat",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=None,
        help="Reference audio for TTS voice style. Default: <model-path>/assets/system_ref_audio.wav",
    )
    parser.add_argument(
        "--language",
        choices=("en", "zh"),
        default="zh",
        help="Assistant language for TTS mode. Default: zh",
    )
    parser.add_argument(
        "--hift-input-len",
        type=int,
        default=100,
        help="Fixed HiFT input length for TTS. Default: 100",
    )
    parser.add_argument(
        "--tts-timesteps",
        type=int,
        default=int(os.getenv("OV_TTS_TIMESTEPS", "3")),
        help="Flow matching denoising steps for TTS token2wav. Lower is faster but may reduce quality. Default: 3",
    )
    parser.add_argument(
        "--token2wav-model-dir",
        type=Path,
        default=Path(os.getenv("OV_TOKEN2WAV_MODEL_DIR")) if os.getenv("OV_TOKEN2WAV_MODEL_DIR") else None,
        help="Directory containing OpenVINO token2wav Flow/HiFT models. Default: --model-path",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_SERVER_NAME", DEFAULT_HOST),
        help=f"Host interface for Gradio. Default: {DEFAULT_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", DEFAULT_PORT)),
        help=f"Port for Gradio. Default: {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the demo in a browser after startup.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the model directory and demo assets, then exit without loading the model.",
    )
    parser.add_argument(
        "--cli-tts",
        default=None,
        help="Run one text-to-speech inference from the CLI, save --output, then exit.",
    )
    parser.add_argument(
        "--tts-task",
        choices=("answer", "read"),
        default="answer",
        help="CLI TTS task: answer the prompt with speech, or read the text aloud. Default: answer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("minicpmo_tts_output.wav"),
        help="Output WAV path for --cli-tts. Default: minicpmo_tts_output.wav",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum generated text tokens for --cli-tts. Default: 256",
    )
    parser.add_argument(
        "--stream-buffer-chunks",
        type=int,
        default=1,
        help="Number of model audio chunks to accumulate before yielding to Gradio streaming. "
        "Higher values reduce ffmpeg/HTTP overhead and smooth playback, but increase first-chunk latency. "
        "Try 2-3 for smoother playback on PTL iGPU. Default: 1 (no accumulation)",
    )
    return parser.parse_args()


def validate_model_dir(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).exists()]
    if missing:
        missing_text = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(f"Model directory is missing required files:\n{missing_text}")

    return model_path


def prepare_demo_assets(model_path: Path) -> None:
    """Expose model-bundled examples to gradio_helper's assets directory."""
    demo_dir = Path(__file__).resolve().parent
    demo_assets = demo_dir / "assets"
    model_assets = model_path / "assets"
    demo_assets.mkdir(exist_ok=True)

    for name in ("system_ref_audio.wav", "highway.png", "fossil.png"):
        source = model_assets / name
        target = demo_assets / name
        if not source.exists() or target.exists():
            continue
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def load_reference_audio(ref_audio_path: Path | None):
    if ref_audio_path is None or not ref_audio_path.exists():
        return None

    import librosa

    ref_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)
    return ref_audio


def generate_speech_file(
    ov_model,
    text: str,
    output_path: Path,
    ref_audio=None,
    language: str = "zh",
    task: str = "answer",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 100,
    repetition_penalty: float = 1.05,
) -> tuple[Path, str]:
    import soundfile as sf
    import torch

    audio_chunks = []
    transcript = ""
    for wav_chunk, text_so_far in iter_speech_chunks(
        ov_model=ov_model,
        text=text,
        ref_audio=ref_audio,
        language=language,
        task=task,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    ):
        transcript = text_so_far
        if wav_chunk is not None:
            audio_chunks.append(wav_chunk.detach().cpu())

    if not audio_chunks:
        raise RuntimeError(f"No audio was generated. Text output: {transcript}")

    waveform = torch.cat(audio_chunks, dim=-1)
    if waveform.ndim > 1:
        waveform = waveform[0]

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform.numpy(), samplerate=24000)
    return output_path, transcript.strip()


def iter_speech_chunks(
    ov_model,
    text: str,
    ref_audio=None,
    language: str = "zh",
    task: str = "answer",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 100,
    repetition_penalty: float = 1.05,
):
    """Yield generated waveform chunks and cumulative transcript text."""

    text = (text or "").strip()
    if not text:
        raise ValueError("Text input is empty.")

    if task == "read":
        user_text = (
            "Read the following text aloud naturally. Do not add explanations or extra content:\n"
            f"{text}"
        )
        sys_mode = "voice_cloning" if ref_audio is not None else "audio_assistant"
    else:
        user_text = text
        sys_mode = "audio_assistant"

    ov_model.reset_session(reset_token2wav_cache=False)
    if ref_audio is not None:
        ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

    session_id = f"tts-{uuid.uuid4().hex}"
    sys_msg = ov_model.get_sys_prompt(
        ref_audio=ref_audio,
        mode=sys_mode,
        language=language,
    )

    ov_model.streaming_prefill(
        session_id=session_id,
        msgs=[sys_msg],
        omni_mode=False,
        is_last_chunk=True,
    )
    ov_model.streaming_prefill(
        session_id=session_id,
        msgs=[{"role": "user", "content": [user_text]}],
        omni_mode=False,
        is_last_chunk=True,
    )

    transcript = ""
    for wav_chunk, text_chunk in ov_model.streaming_generate(
        session_id=session_id,
        generate_audio=True,
        use_tts_template=True,
        enable_thinking=False,
        do_sample=temperature > 0,
        max_new_tokens=int(max_new_tokens),
        temperature=max(float(temperature), 0.01),
        top_p=float(top_p),
        top_k=int(top_k),
        repetition_penalty=float(repetition_penalty),
        length_penalty=1.1,
    ):
        if text_chunk:
            transcript += text_chunk
        if wav_chunk is not None:
            yield wav_chunk, transcript.strip()


def make_tts_demo(ov_model, ref_audio_path: Path | None, language: str, stream_buffer_chunks: int = 1):
    import time

    import gradio as gr
    import numpy as np
    import soundfile as sf
    import torch

    inference_lock = threading.Lock()
    ref_audio = load_reference_audio(ref_audio_path)
    if ref_audio is not None:
        ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

    def synthesize(
        text,
        task,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    ):
        text = (text or "").strip()
        if not text:
            return None, "Please enter text first."

        with inference_lock:
            try:
                output_path = Path(tempfile.gettempdir()) / f"minicpmo_tts_{uuid.uuid4().hex}.wav"
                result_path, transcript = generate_speech_file(
                    ov_model=ov_model,
                    text=text,
                    output_path=output_path,
                    ref_audio=ref_audio,
                    language=language,
                    task="read" if task == "Read text aloud" else "answer",
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                )
                return str(result_path), transcript
            except Exception:
                return None, f"Generation error:\n{traceback.format_exc()}"

    def stream_synthesize(
        text,
        task,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
    ):
        text = (text or "").strip()
        if not text:
            yield None, None, "Please enter text first."
            return

        with inference_lock:
            audio_chunks = []
            buffer = []  # accumulate small chunks before yielding
            transcript = ""
            output_path = Path(tempfile.gettempdir()) / f"minicpmo_tts_stream_{uuid.uuid4().hex}.wav"
            chunk_idx = 0
            t_prev = time.perf_counter()
            try:
                for wav_chunk, transcript in iter_speech_chunks(
                    ov_model=ov_model,
                    text=text,
                    ref_audio=ref_audio,
                    language=language,
                    task="read" if task == "Read text aloud" else "answer",
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    repetition_penalty=float(repetition_penalty),
                ):
                    if wav_chunk is None:
                        continue
                    wav_chunk = wav_chunk.detach().cpu()
                    audio_chunks.append(wav_chunk)
                    buffer.append(wav_chunk)

                    # Timing diagnostics
                    t_now = time.perf_counter()
                    chunk_samples = wav_chunk.numel()
                    chunk_dur_ms = chunk_samples / 24000 * 1000
                    gen_interval_ms = (t_now - t_prev) * 1000
                    chunk_idx += 1
                    print(
                        f"  [stream] chunk {chunk_idx}: "
                        f"gen {gen_interval_ms:.0f}ms, audio {chunk_dur_ms:.0f}ms, "
                        f"RTF {gen_interval_ms / max(chunk_dur_ms, 1):.3f}"
                    )
                    t_prev = t_now

                    # Yield accumulated buffer when enough chunks collected
                    if len(buffer) >= stream_buffer_chunks:
                        merged = torch.cat(buffer, dim=-1)
                        merged = merged[0] if merged.ndim > 1 else merged
                        buffer.clear()
                        yield (24000, merged.numpy().astype(np.float32)), None, transcript

                # Flush remaining buffer
                if buffer:
                    merged = torch.cat(buffer, dim=-1)
                    merged = merged[0] if merged.ndim > 1 else merged
                    buffer.clear()
                    yield (24000, merged.numpy().astype(np.float32)), None, transcript

                if not audio_chunks:
                    yield None, None, f"No audio was generated. Text output: {transcript}"
                    return

                waveform = torch.cat(audio_chunks, dim=-1)
                if waveform.ndim > 1:
                    waveform = waveform[0]
                sf.write(str(output_path), waveform.numpy(), samplerate=24000)
                yield None, str(output_path), transcript
            except Exception:
                yield None, None, f"Generation error:\n{traceback.format_exc()}"

    with gr.Blocks(title="MiniCPM-o 4.5 TTS") as demo:
        gr.Markdown("# MiniCPM-o 4.5 Text To Speech")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    label="Text input",
                    value="请用自然的中文语音介绍一下 OpenVINO 是什么。",
                    lines=5,
                )
                task = gr.Radio(
                    choices=("Answer with speech", "Read text aloud"),
                    value="Answer with speech",
                    label="Task",
                )
                submit = gr.Button("Generate Speech", variant="primary")
                stream_submit = gr.Button("Stream Speech")
            with gr.Column():
                stream_audio = gr.Audio(
                    label="Streaming preview",
                    type="numpy",
                    streaming=True,
                    autoplay=True,
                )
                audio = gr.Audio(label="Generated speech file", type="filepath")
                transcript = gr.Textbox(label="Generated text", lines=6)

        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                max_new_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max tokens")
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            with gr.Row():
                top_p = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Top-P")
                top_k = gr.Slider(1, 200, value=100, step=1, label="Top-K")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.05, step=0.01, label="Repetition penalty")

        submit.click(
            synthesize,
            [text, task, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            [audio, transcript],
        )
        stream_submit.click(
            stream_synthesize,
            [text, task, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
            [stream_audio, audio, transcript],
        )

    return demo


def run_ws_tts_server(ov_model, ref_audio_path: Path | None, language: str, host: str, port: int):
    """Run a WebSocket TTS server with raw PCM streaming (no origin checking).

    HTTP static server on `port`, WebSocket on `port + 1`.
    """
    import asyncio
    import http.server
    import json
    import time

    import numpy as np
    import websockets

    inference_lock = threading.Lock()
    ref_audio = load_reference_audio(ref_audio_path)
    if ref_audio is not None:
        ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

    demo_dir = Path(__file__).resolve().parent
    ws_port = port + 1

    # --- HTTP static file server (serves HTML + worklet.js) ---
    class TTSHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(demo_dir), **kwargs)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.path = "/ws_tts_player.html"
            elif self.path == "/worklet.js":
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.end_headers()
                self.wfile.write(WORKLET_JS)
                return
            super().do_GET()

        def log_message(self, format, *args):
            pass  # suppress HTTP logs

    WORKLET_JS = b"""\
class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(0);
    this.port.onmessage = (e) => {
      if (e.data.type === 'audio') {
        const old = this.buffer;
        this.buffer = new Float32Array(old.length + e.data.samples.length);
        this.buffer.set(old);
        this.buffer.set(e.data.samples, old.length);
      } else if (e.data.type === 'clear') {
        this.buffer = new Float32Array(0);
      }
    };
  }
  process(inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;
    const needed = output.length;
    if (this.buffer.length >= needed) {
      output.set(this.buffer.subarray(0, needed));
      this.buffer = this.buffer.slice(needed);
    } else {
      output.fill(0);
    }
    this.port.postMessage({ bufferSamples: this.buffer.length });
    return true;
  }
}
registerProcessor('pcm-player-processor', PCMPlayerProcessor);
"""

    # --- WebSocket handler ---
    async def handle_tts(websocket):
        """Handle a WebSocket TTS session."""
        try:
            data = await websocket.recv()
            msg = json.loads(data)
            text = (msg.get("text") or "").strip()
            task = msg.get("task", "answer")

            if not text:
                await websocket.send(json.dumps({"error": "Empty text"}))
                return

            with inference_lock:
                chunk_idx = 0
                t_start = time.perf_counter()
                t_prev = t_start

                for wav_chunk, transcript in iter_speech_chunks(
                    ov_model=ov_model,
                    text=text,
                    ref_audio=ref_audio,
                    language=language,
                    task=task,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=100,
                    repetition_penalty=1.05,
                ):
                    if wav_chunk is None:
                        continue

                    chunk_idx += 1
                    t_now = time.perf_counter()
                    samples = wav_chunk.detach().cpu()
                    if samples.ndim > 1:
                        samples = samples[0]
                    pcm = samples.numpy().astype(np.float32)

                    # Timing log
                    chunk_dur_ms = len(pcm) / 24000 * 1000
                    gen_ms = (t_now - t_prev) * 1000
                    print(
                        f"  [ws-tts] chunk {chunk_idx}: "
                        f"gen {gen_ms:.0f}ms, audio {chunk_dur_ms:.0f}ms, "
                        f"RTF {gen_ms / max(chunk_dur_ms, 1):.3f}"
                    )
                    t_prev = t_now

                    # Send binary PCM frame
                    await websocket.send(pcm.tobytes())
                    # Send transcript update
                    await websocket.send(json.dumps({"transcript": transcript}))

                total_ms = (time.perf_counter() - t_start) * 1000
                print(f"  [ws-tts] done: {chunk_idx} chunks in {total_ms:.0f}ms")
                await websocket.send(json.dumps({"done": True, "transcript": transcript}))

        except websockets.exceptions.ConnectionClosed:
            print("  [ws-tts] client disconnected")
        except Exception as exc:
            try:
                await websocket.send(json.dumps({"error": str(exc)}))
            except Exception:
                pass
            traceback.print_exc()

    # --- Start servers ---
    def start_http():
        httpd = http.server.HTTPServer((host, port), TTSHTTPHandler)
        httpd.serve_forever()

    http_thread = threading.Thread(target=start_http, daemon=True)
    http_thread.start()

    print(f"Starting WebSocket TTS server:")
    print(f"  HTML page:  http://{host}:{port}")
    print(f"  WebSocket:  ws://{host}:{ws_port}")
    print(f"  Open http://{host}:{port} in browser")

    async def serve():
        async with websockets.serve(handle_tts, host, ws_port, origins=None):
            await asyncio.Future()  # run forever

    asyncio.run(serve())


def run_ws_omni_server(ov_model, ref_audio_path: Path | None, language: str, host: str, port: int):
    """Run a WebSocket omni-live server: video+audio input → streaming speech+text output.

    HTTP static server on `port`, WebSocket on `port + 1`.
    """
    import asyncio
    import base64
    import http.server
    import io
    import json
    import time

    import numpy as np
    import websockets
    from PIL import Image

    inference_lock = threading.Lock()
    ref_audio = load_reference_audio(ref_audio_path)
    if ref_audio is not None:
        ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

    demo_dir = Path(__file__).resolve().parent
    ws_port = port + 1

    keyframe_resize_filter = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

    def _frame_signature(frame):
        small = frame.convert("L").resize((96, 54), keyframe_resize_filter)
        return np.asarray(small, dtype=np.float32) / 255.0

    def _frame_change_metrics(previous, current):
        if previous is None or current is None:
            return 1.0, 1.0
        diff = np.abs(current - previous)
        mean_delta = float(np.mean(diff))
        changed_ratio = float(np.mean(diff > 0.10))
        return mean_delta, changed_ratio

    def _is_effective_keyframe(previous, current):
        mean_delta, changed_ratio = _frame_change_metrics(previous, current)
        return mean_delta >= 0.018 or changed_ratio >= 0.015, mean_delta, changed_ratio

    def _classify_vision_intent(text, interaction_mode, auto_vision, current_prompt):
        """Classify whether the user wants continuous camera understanding."""
        normalized = (text or "").strip().lower()
        if not normalized:
            return {
                "intent": "NO_TEXT",
                "action": "none",
                "vision_prompt": current_prompt,
                "reason": "empty text",
            }

        stop_terms = (
            "不用看", "不要看", "别看", "停止看", "停止观察", "停止理解", "关闭视觉",
            "关掉视觉", "关掉摄像头理解", "先不看", "别管画面", "stop watching",
            "结束对话", "终止对话", "结束demo", "结束演示", "stop looking", "stop vision",
            "turn off vision",
        )
        pause_terms = (
            "暂停观察", "暂停看", "先暂停", "pause watching", "pause vision",
        )
        resume_terms = (
            "继续看", "继续观察", "接着看", "恢复观察", "resume watching", "resume vision",
        )
        start_terms = (
            "持续观察", "持续看", "一直看", "一直观察", "帮我看着", "帮我盯着",
            "盯着", "有变化告诉我", "变化了告诉我", "实时看", "持续理解",
            "按顺序识别", "依次识别", "逐个回答", "逐一回答", "逐个识别",
            "一个一个识别", "挨个识别", "按顺序回答", "依次回答",
            "计算数字的总和", "数字的总和", "累计求和", "累加",
            "keep watching", "watch continuously", "monitor this", "tell me if it changes",
            "identify in order", "answer one by one", "recognize one by one",
        )
        visual_terms = (
            "画面", "图片", "图像", "摄像头", "镜头", "视频", "照片", "这张图",
            "这个", "这里", "看一下", "看下", "穿搭", "颜色", "物品", "手势",
            "image", "picture", "camera", "video", "frame", "look at", "see this",
        )

        if any(term in normalized for term in stop_terms):
            return {
                "intent": "STOP_VISUAL_TRACKING",
                "action": "stop_continuous_vision",
                "vision_prompt": "",
                "reason": "matched stop term",
            }
        if any(term in normalized for term in pause_terms):
            return {
                "intent": "PAUSE_VISUAL_TRACKING",
                "action": "pause_continuous_vision",
                "vision_prompt": current_prompt,
                "reason": "matched pause term",
            }
        if any(term in normalized for term in resume_terms):
            return {
                "intent": "RESUME_VISUAL_TRACKING",
                "action": "resume_continuous_vision",
                "vision_prompt": current_prompt or text.strip(),
                "reason": "matched resume term",
            }
        if any(term in normalized for term in start_terms):
            return {
                "intent": "START_VISUAL_TRACKING",
                "action": "start_continuous_vision",
                "vision_prompt": text.strip(),
                "reason": "matched continuous vision term",
            }
        if interaction_mode == "continuous" and auto_vision and current_prompt and any(term in normalized for term in visual_terms):
            return {
                "intent": "UPDATE_VISUAL_GOAL",
                "action": "update_vision_prompt",
                "vision_prompt": text.strip(),
                "reason": "visual request while continuous vision is active",
            }
        if any(term in normalized for term in visual_terms):
            return {
                "intent": "ONE_SHOT_VISUAL_QUERY",
                "action": "single_frame_answer",
                "vision_prompt": text.strip(),
                "reason": "visual request without continuous tracking",
            }
        return {
            "intent": "NON_VISUAL_CHAT",
            "action": "answer_without_changing_vision",
            "vision_prompt": current_prompt,
            "reason": "no visual control terms matched",
        }

    def _is_sequence_counting_prompt(prompt):
        normalized = (prompt or "").strip().lower()
        sequence_terms = (
            "依次", "按顺序", "逐个", "逐一", "一个一个", "挨个",
            "in order", "one by one",
        )
        counting_terms = (
            "总和", "求和", "累加", "累计", "计算", "加起来", "手势",
            "sum", "total", "add up", "gesture",
        )
        return any(term in normalized for term in sequence_terms) and any(term in normalized for term in counting_terms)

    async def _send_intent_log(websocket, decision):
        prompt = decision.get("vision_prompt") or ""
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        message = (
            "Intent recognized: "
            f"{decision.get('intent')} "
            f"(action={decision.get('action')}, reason={decision.get('reason')}"
        )
        if prompt:
            message += f", vision_prompt={prompt}"
        message += ")"
        await websocket.send(json.dumps({"info": message}, ensure_ascii=False))

    # --- HTTP server ---
    class OmniHTTPHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(demo_dir), **kwargs)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.path = "/ws_omni_live.html"
            super().do_GET()

        def log_message(self, format, *args):
            pass

    # --- WebSocket handler ---
    async def handle_omni(websocket):
        """Handle an omni-live WebSocket session with full-duplex voice interaction.

        The duplex loop:
        1. Browser sends mic audio chunks (binary) + video frames (JSON)
        2. Server accumulates ~1s of audio, then feeds to duplex model
        3. Model decides: listen (wait for more) or speak (generate response)
        4. When speaking, streams PCM audio + text back to browser
        """
        latest_frame = None  # Most recent video frame (PIL Image)
        recent_frames = deque(maxlen=4)
        audio_queue = asyncio.Queue()  # Mic audio chunks
        query_queue = asyncio.Queue()  # Browser ASR/manual text queries
        stop_event = asyncio.Event()
        interrupt_event = threading.Event()
        duplex_mode = False  # Whether to use full-duplex loop
        backend_asr = False
        auto_vision = False
        interaction_mode = "normal"
        vision_interval = 3.0
        vision_user_prompt = ""
        vision_prompt_version = 0
        voice_busy = False
        last_voice_activity = 0.0
        assistant_busy_until = 0.0
        gesture_window_seconds = 5.0

        def remember_frame(frame):
            recent_frames.append((time.perf_counter(), frame.copy()))

        def get_recent_video_frames():
            if not recent_frames:
                return [], 0.0
            now = time.perf_counter()
            selected = [(ts, frame.copy()) for ts, frame in recent_frames if now - ts <= gesture_window_seconds]
            if not selected:
                return [], 0.0
            frames = [frame for _ts, frame in selected[-4:]]
            span = max(0.0, selected[-1][0] - selected[0][0])
            return frames, span

        async def apply_vision_intent(decision):
            nonlocal interaction_mode, auto_vision, vision_user_prompt, vision_prompt_version

            action = decision.get("action")
            prompt = (decision.get("vision_prompt") or "").strip()

            if action in ("start_continuous_vision", "resume_continuous_vision"):
                interaction_mode = "continuous"
                auto_vision = True
                if prompt and prompt != vision_user_prompt:
                    vision_user_prompt = prompt
                    vision_prompt_version += 1
                await websocket.send(json.dumps({
                    "state": {
                        "interaction_mode": interaction_mode,
                        "auto_vision": auto_vision,
                        "vision_prompt": vision_user_prompt or prompt,
                    }
                }, ensure_ascii=False))
                await websocket.send(json.dumps({"info": f"Continuous vision enabled: {vision_user_prompt or prompt}"}, ensure_ascii=False))
                return

            if action == "update_vision_prompt" and prompt:
                if prompt != vision_user_prompt:
                    vision_user_prompt = prompt
                    vision_prompt_version += 1
                await websocket.send(json.dumps({
                    "state": {
                        "interaction_mode": interaction_mode,
                        "auto_vision": auto_vision,
                        "vision_prompt": vision_user_prompt,
                    }
                }, ensure_ascii=False))
                await websocket.send(json.dumps({"info": f"Vision prompt updated: {vision_user_prompt}"}, ensure_ascii=False))
                return

            if action in ("stop_continuous_vision", "pause_continuous_vision"):
                auto_vision = False
                if action == "stop_continuous_vision":
                    vision_user_prompt = ""
                    vision_prompt_version += 1
                await websocket.send(json.dumps({
                    "state": {
                        "interaction_mode": interaction_mode,
                        "auto_vision": auto_vision,
                        "vision_prompt": vision_user_prompt,
                    }
                }, ensure_ascii=False))
                await websocket.send(json.dumps({"info": "Continuous vision disabled."}))

        async def receive_loop():
            """Receive messages from browser and route to queues."""
            nonlocal latest_frame, duplex_mode, backend_asr, auto_vision, interaction_mode, vision_interval, vision_user_prompt, vision_prompt_version
            try:
                async for raw_msg in websocket:
                    if stop_event.is_set():
                        break
                    if isinstance(raw_msg, bytes):
                        # Binary: microphone audio
                        if len(raw_msg) > 4 and raw_msg[:4] == b'MIC\x00':
                            pcm = np.frombuffer(raw_msg[4:], dtype=np.float32)
                            if not hasattr(receive_loop, '_mic_log_count'):
                                receive_loop._mic_log_count = 0
                            receive_loop._mic_log_count += 1
                            if receive_loop._mic_log_count <= 5:
                                rms_val = float(np.sqrt(np.mean(pcm ** 2))) if len(pcm) > 0 else 0
                                print(f"  [recv] mic packet #{receive_loop._mic_log_count}: {len(raw_msg)} bytes, {len(pcm)} samples, rms={rms_val:.6f}, first5={pcm[:5].tolist()}")
                            await audio_queue.put(pcm)
                        else:
                            print(f"  [recv] unknown binary: {len(raw_msg)} bytes, header={raw_msg[:8].hex()}")
                        continue

                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type", "")

                    if msg_type == "config":
                        mode = msg.get("mode", "omni")
                        duplex_mode = (mode == "duplex")
                        requested_interaction = msg.get("interaction_mode", "normal")
                        interaction_mode = "continuous" if requested_interaction == "continuous" else "normal"
                        backend_asr = bool(msg.get("backend_asr", False))
                        auto_vision = interaction_mode == "continuous" and bool(msg.get("auto_vision", False))
                        if interaction_mode == "normal" and vision_user_prompt:
                            vision_user_prompt = ""
                            vision_prompt_version += 1
                        try:
                            vision_interval = max(1.0, float(msg.get("vision_interval", 3.0)))
                        except (TypeError, ValueError):
                            vision_interval = 3.0
                        await websocket.send(json.dumps({"info": f"Session started, mode: {mode}, interaction: {interaction_mode}"}))
                        if backend_asr:
                            await websocket.send(json.dumps({"info": "Backend speech ASR enabled."}))
                        if auto_vision:
                            await websocket.send(json.dumps({"info": f"Continuous keyframe description enabled ({vision_interval:.0f}s interval)."}))

                    elif msg_type == "frame":
                        img_data = msg.get("image", "")
                        if img_data and "," in img_data:
                            img_data = img_data.split(",", 1)[1]
                        if img_data:
                            try:
                                img_bytes = base64.b64decode(img_data)
                                latest_frame = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                remember_frame(latest_frame)
                            except Exception as e:
                                print(f"  [omni] frame decode error: {e}")

                    elif msg_type == "query":
                        # Manual query: put a sentinel to trigger inference
                        text = (msg.get("text") or "").strip()
                        img_data = msg.get("image", "")
                        audio_data_url = msg.get("audio", "")
                        if duplex_mode:
                            await query_queue.put(("QUERY", text, img_data, audio_data_url))
                        else:
                            await audio_queue.put(("QUERY", text, img_data, audio_data_url))

                    elif msg_type == "interrupt":
                        interrupt_event.set()
                        await websocket.send(json.dumps({"info": "Assistant response interrupted by user speech."}))

                    elif msg_type == "stop":
                        stop_event.set()
                        break

            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                stop_event.set()

        async def duplex_loop():
            """Continuous microphone loop: stream audio and answer browser-ASR text queries."""
            nonlocal latest_frame, voice_busy, last_voice_activity, vision_user_prompt, vision_prompt_version, assistant_busy_until, interaction_mode

            await websocket.send(json.dumps({"info": "Voice loop started. Speak into microphone."}))
            print("  [omni-voice] loop started")

            chunk_samples = int(0.5 * 16000)
            speech_threshold = 0.008
            silence_threshold = 0.004
            end_silence_samples = int(0.8 * 16000)
            min_speech_samples = int(0.6 * 16000)
            max_segment_samples = int(12.0 * 16000)

            audio_acc = np.zeros(0, dtype=np.float32)
            speech_parts = []
            speech_samples = 0
            silence_samples = 0
            speech_active = False
            chunk_idx = 0

            async def handle_query(item):
                nonlocal voice_busy, last_voice_activity, vision_user_prompt, vision_prompt_version, assistant_busy_until
                _, text, _img_data, _audio_data_url = item
                content = []
                if latest_frame is not None:
                    content.append(latest_frame)
                if text:
                    intent_decision = _classify_vision_intent(text, interaction_mode, auto_vision, vision_user_prompt)
                    await _send_intent_log(websocket, intent_decision)
                    await apply_vision_intent(intent_decision)
                    content.append(text)
                    await websocket.send(json.dumps({"role": "user", "text": text}))
                if not content:
                    return
                try:
                    interrupt_event.clear()
                    voice_busy = True
                    last_voice_activity = time.perf_counter()
                    await websocket.send(json.dumps({"info": "Generating assistant response..."}))
                    with inference_lock:
                        _transcript, audio_ms = await _run_inference(websocket, ov_model, content, ref_audio, language, interrupt_event)
                    assistant_busy_until = max(assistant_busy_until, time.perf_counter() + audio_ms / 1000.0 + 0.8)
                except Exception as exc:
                    print(f"  [omni-voice] query inference error: {exc}")
                    traceback.print_exc()
                    await websocket.send(json.dumps({"error": str(exc)}))
                finally:
                    last_voice_activity = time.perf_counter()
                    voice_busy = False

            while not stop_event.is_set():
                try:
                    pending_query = query_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pending_query = None
                if pending_query is not None:
                    await handle_query(pending_query)
                    continue

                try:
                    pcm = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                if isinstance(pcm, tuple) and pcm[0] == "QUERY":
                    await handle_query(pcm)
                    continue

                audio_acc = np.concatenate([audio_acc, pcm])
                while len(audio_acc) >= chunk_samples:
                    audio_chunk = audio_acc[:chunk_samples]
                    audio_acc = audio_acc[chunk_samples:]
                    chunk_idx += 1
                    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

                    if chunk_idx <= 5 or chunk_idx % 10 == 0:
                        print(f"  [omni-voice] chunk {chunk_idx}: rms={rms:.5f}, active={speech_active}")

                    if rms >= speech_threshold:
                        if not speech_active:
                            voice_busy = True
                            last_voice_activity = time.perf_counter()
                            await websocket.send(json.dumps({"info": "Speech detected, listening..."}))
                            print(f"  [omni-voice] speech start at chunk {chunk_idx}")
                        speech_active = True
                        silence_samples = 0
                        speech_parts.append(audio_chunk)
                        speech_samples += len(audio_chunk)
                    elif speech_active:
                        speech_parts.append(audio_chunk)
                        speech_samples += len(audio_chunk)
                        if rms <= silence_threshold:
                            silence_samples += len(audio_chunk)
                        else:
                            silence_samples = 0

                    should_finalize = speech_active and (
                        silence_samples >= end_silence_samples or speech_samples >= max_segment_samples
                    )
                    if not should_finalize:
                        continue

                    speech_audio = np.concatenate(speech_parts) if speech_parts else np.zeros(0, dtype=np.float32)
                    speech_parts.clear()
                    speech_active = False
                    silence_samples = 0
                    speech_samples = 0

                    if not backend_asr:
                        await websocket.send(json.dumps({"info": "Speech ended. Waiting for browser ASR final text..."}))
                        print("  [omni-voice] speech ended; waiting for browser ASR query")
                        voice_busy = False
                        continue

                    if len(speech_audio) < min_speech_samples:
                        continue

                    segment_rms = float(np.sqrt(np.mean(speech_audio ** 2)))
                    print(f"  [omni-voice] backend ASR segment: {len(speech_audio)/16000:.2f}s, rms={segment_rms:.5f}")
                    await websocket.send(json.dumps({"info": "Recognizing speech on backend..."}))

                    try:
                        voice_busy = True
                        last_voice_activity = time.perf_counter()
                        with inference_lock:
                            user_text = _run_asr_transcription(ov_model, speech_audio, language)
                    except Exception as exc:
                        print(f"  [omni-voice] backend ASR error: {exc}")
                        traceback.print_exc()
                        await websocket.send(json.dumps({"error": f"Backend ASR failed: {exc}"}))
                        voice_busy = False
                        continue

                    user_text = (user_text or "").strip()
                    if not user_text:
                        await websocket.send(json.dumps({"info": "Backend ASR returned empty text."}))
                        voice_busy = False
                        continue

                    print(f"  [omni-voice] backend ASR: {user_text}")
                    await handle_query(("QUERY", user_text, "", ""))

        async def vision_loop():
            """Continuously describe the latest camera frame when auto vision is enabled."""
            nonlocal latest_frame, vision_user_prompt, vision_prompt_version, assistant_busy_until, interaction_mode

            last_run = 0.0
            previous_summary = ""
            previous_prompt_version = vision_prompt_version
            keyframe_signature = None

            while not stop_event.is_set():
                await asyncio.sleep(0.5)
                if interaction_mode != "continuous" or not auto_vision or latest_frame is None:
                    continue
                if not vision_user_prompt:
                    continue
                if voice_busy or (time.perf_counter() - last_voice_activity) < 2.0:
                    continue
                now = time.perf_counter()
                if now < assistant_busy_until:
                    continue
                if now - last_run < vision_interval:
                    continue
                if not query_queue.empty():
                    continue

                frame = latest_frame.copy()
                video_frames, video_span = get_recent_video_frames()
                if not video_frames:
                    video_frames = [frame]
                    video_span = 0.0
                last_run = now
                if previous_prompt_version != vision_prompt_version:
                    previous_summary = ""
                    previous_prompt_version = vision_prompt_version
                    keyframe_signature = None

                frame_signature = _frame_signature(frame)
                is_keyframe, mean_delta, changed_ratio = _is_effective_keyframe(keyframe_signature, frame_signature)
                if not is_keyframe:
                    print(
                        "  [omni-vision] skipped similar frame: "
                        f"mean_delta={mean_delta:.4f}, changed={changed_ratio:.3f}"
                    )
                    continue

                active_prompt = vision_user_prompt
                frame_count = len(video_frames)
                content_prompt = (
                    f"请按照用户的展示要求理解这组按时间顺序排列的连续摄像头画面：{active_prompt}。"
                    f"这组画面共有{frame_count}帧，覆盖最近约{video_span:.1f}秒；第一帧最早，最后一帧最新。"
                    "请结合多帧上下文判断动作、物品或数字的变化，优先描述最后一帧相对于之前帧的新增变化；"
                    "不要提到你是从图片中看到的。"
                )
                if _is_sequence_counting_prompt(active_prompt):
                    content_prompt += (
                        "这是一个连续手势数字识别和累计求和任务。"
                        f"默认以最近{gesture_window_seconds:.0f}秒作为一个手势完成判断窗口："
                        "如果同一数字或手势在这个窗口内保持稳定，则认为当前手势已完成，可以识别并计入；"
                        "每次只在手势或数字发生明确变化时，识别当前新数字，并基于历史结果更新累计总和；"
                        "如果当前画面和上一次相同，不要重复累加。"
                        "回答格式请尽量简洁：当前数字：N；累计总和：S。"
                    )
                if previous_summary:
                    content_prompt += f" 上一次描述是：{previous_summary}。如果画面相似，请只补充变化。"

                try:
                    await websocket.send(json.dumps({"info": "Generating assistant response for camera..."}))
                    print(
                        "  [omni-vision] describing keyframe: "
                        f"frames={frame_count}, window={video_span:.1f}s, "
                        f"mean_delta={mean_delta:.4f}, changed={changed_ratio:.3f}"
                    )
                    with inference_lock:
                        interrupt_event.clear()
                        transcript, audio_ms = await _run_inference(
                            websocket,
                            ov_model,
                            [*video_frames, content_prompt],
                            ref_audio,
                            language,
                            interrupt_event,
                            max_new_tokens=128,
                        )
                    assistant_busy_until = max(assistant_busy_until, time.perf_counter() + audio_ms / 1000.0 + 0.8)
                    if transcript:
                        previous_summary = transcript[-200:]
                        keyframe_signature = frame_signature
                except Exception as exc:
                    print(f"  [omni-vision] inference error: {exc}")
                    traceback.print_exc()
                    try:
                        await websocket.send(json.dumps({"error": str(exc)}))
                    except Exception:
                        pass

        async def simplex_loop():
            """Simplex processing: wait for explicit queries."""
            nonlocal latest_frame, vision_user_prompt, vision_prompt_version, assistant_busy_until, interaction_mode
            audio_buffer_local = []

            while not stop_event.is_set():
                try:
                    item = await asyncio.wait_for(audio_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                if isinstance(item, tuple) and item[0] == "QUERY":
                    _, text, img_data, audio_data_url = item

                    # Decode image
                    query_frame = latest_frame
                    if img_data and "," in img_data:
                        img_data_b64 = img_data.split(",", 1)[1]
                        try:
                            img_bytes = base64.b64decode(img_data_b64)
                            query_frame = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        except Exception:
                            pass

                    # Combine mic audio
                    query_audio = None
                    if audio_buffer_local:
                        query_audio = np.concatenate(audio_buffer_local)
                        audio_buffer_local.clear()

                    # Decode uploaded audio file
                    if audio_data_url and "," in audio_data_url:
                        try:
                            import librosa
                            audio_b64 = audio_data_url.split(",", 1)[1]
                            audio_bytes = base64.b64decode(audio_b64)
                            audio_wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
                            query_audio = np.concatenate([query_audio, audio_wav]) if query_audio is not None else audio_wav
                        except Exception as e:
                            print(f"  [omni] audio decode error: {e}")

                    # Build content
                    content = []
                    if query_frame is not None:
                        content.append(query_frame)
                    if query_audio is not None and len(query_audio) > 1600:
                        content.append((query_audio, 16000))
                    if text:
                        intent_decision = _classify_vision_intent(text, interaction_mode, auto_vision, vision_user_prompt)
                        await _send_intent_log(websocket, intent_decision)
                        await apply_vision_intent(intent_decision)
                        content.append(text)

                    if not content:
                        await websocket.send(json.dumps({"error": "No input provided"}))
                        continue

                    with inference_lock:
                        interrupt_event.clear()
                        _transcript, audio_ms = await _run_inference(websocket, ov_model, content, ref_audio, language, interrupt_event)
                    assistant_busy_until = max(assistant_busy_until, time.perf_counter() + audio_ms / 1000.0 + 0.8)
                else:
                    # Regular audio chunk — buffer for next query
                    audio_buffer_local.append(item)

        try:
            # Run receive loop concurrently with processing loop
            recv_task = asyncio.create_task(receive_loop())

            # Wait a moment for config message
            await asyncio.sleep(0.3)

            if duplex_mode:
                proc_task = asyncio.create_task(duplex_loop())
                vision_task = asyncio.create_task(vision_loop())
            else:
                proc_task = asyncio.create_task(simplex_loop())
                vision_task = asyncio.create_task(vision_loop())

            # Wait for either to finish
            done, pending = await asyncio.wait(
                [recv_task, proc_task, vision_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            stop_event.set()
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        except websockets.exceptions.ConnectionClosed:
            print("  [omni] client disconnected")
        except Exception as exc:
            try:
                await websocket.send(json.dumps({"error": str(exc)}))
            except Exception:
                pass
            traceback.print_exc()

    def _run_asr_transcription(model, audio_waveform, language):
        """Transcribe a 16 kHz speech segment using MiniCPM-o text-only generation."""
        if audio_waveform is None or len(audio_waveform) == 0:
            return ""

        model.reset_session(reset_token2wav_cache=True)
        prompt = (
            "请仔细听这段音频片段，并将其内容逐字记录。"
            if language == "zh"
            else "Please listen to the audio snippet carefully and transcribe the content."
        )
        transcript = model.chat(
            msgs=[{"role": "user", "content": [prompt, audio_waveform]}],
            omni_mode=False,
            stream_input=False,
            generate_audio=False,
            use_tts_template=False,
            enable_thinking=False,
            do_sample=False,
            max_new_tokens=128,
            temperature=0.01,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        return str(transcript).strip()

    async def _run_inference(websocket, model, content, ref_audio, language, interrupt_event=None, max_new_tokens=512):
        """Run simplex inference and stream back audio+text."""
        import time as _time

        t_start = _time.perf_counter()

        # Reset session
        model.reset_session(reset_token2wav_cache=False)
        if ref_audio is not None:
            model.init_token2wav_cache(prompt_speech_16k=ref_audio)

        session_id = f"omni-{uuid.uuid4().hex}"

        # System prompt
        sys_msg = model.get_sys_prompt(
            ref_audio=ref_audio,
            mode="audio_assistant",
            language=language,
        )
        model.streaming_prefill(
            session_id=session_id,
            msgs=[sys_msg],
            omni_mode=False,
            is_last_chunk=True,
        )

        # User message
        model.streaming_prefill(
            session_id=session_id,
            msgs=[{"role": "user", "content": content}],
            omni_mode=True,
            is_last_chunk=True,
        )

        transcript = ""
        chunk_idx = 0
        total_audio_ms = 0.0
        t_prev = _time.perf_counter()
        t_generate_start = t_prev
        first_text_ms = None
        first_output_ms = None

        loop = asyncio.get_running_loop()
        chunk_queue = asyncio.Queue()

        def _generate_worker():
            try:
                for wav_chunk, text_chunk in model.streaming_generate(
                    session_id=session_id,
                    generate_audio=True,
                    use_tts_template=True,
                    enable_thinking=False,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=100,
                    repetition_penalty=1.05,
                    length_penalty=1.1,
                    interrupt_event=interrupt_event,
                ):
                    if interrupt_event is not None and interrupt_event.is_set():
                        break
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, ("chunk", wav_chunk, text_chunk, None))
            except Exception as exc:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("error", None, None, exc))
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, ("done", None, None, None))

        worker = threading.Thread(target=_generate_worker, daemon=True)
        worker.start()

        while True:
            kind, wav_chunk, text_chunk, error = await chunk_queue.get()
            if kind == "error":
                raise error
            if kind == "done":
                break
            if interrupt_event is not None and interrupt_event.is_set():
                break

            if text_chunk:
                if first_text_ms is None:
                    first_text_ms = (_time.perf_counter() - t_start) * 1000
                transcript += text_chunk

            if wav_chunk is not None:
                chunk_idx += 1
                t_now = _time.perf_counter()
                if first_output_ms is None:
                    first_output_ms = (t_now - t_start) * 1000
                samples = wav_chunk.detach().cpu()
                if samples.ndim > 1:
                    samples = samples[0]
                pcm = samples.numpy().astype(np.float32)

                chunk_dur_ms = len(pcm) / 24000 * 1000
                total_audio_ms += chunk_dur_ms
                gen_ms = (t_now - t_prev) * 1000
                print(
                    f"  [omni] chunk {chunk_idx}: "
                    f"gen {gen_ms:.0f}ms, audio {chunk_dur_ms:.0f}ms, "
                    f"RTF {gen_ms / max(chunk_dur_ms, 1):.3f}"
                )
                t_prev = t_now

                # Send PCM audio
                await websocket.send(pcm.tobytes())

            # Send transcript update periodically
            if transcript and (wav_chunk is not None or text_chunk):
                await websocket.send(json.dumps({"transcript": transcript.strip()}))

        total_ms = (_time.perf_counter() - t_start) * 1000
        generation_ms = max((_time.perf_counter() - t_generate_start) * 1000, 1.0)
        try:
            token_count = len(model.tokenizer.encode(transcript)) if transcript else 0
        except Exception:
            token_count = len(transcript)
        metrics = {
            "ttft_ms": round(first_text_ms if first_text_ms is not None else first_output_ms or total_ms),
            "tps": round(token_count / (generation_ms / 1000), 2) if generation_ms > 0 else 0.0,
            "tts_rtf": round((generation_ms / max(total_audio_ms, 1.0)), 3) if total_audio_ms > 0 else None,
            "tokens": token_count,
        }

        if interrupt_event is not None and interrupt_event.is_set():
            print("  [omni] generation interrupted")
            await asyncio.to_thread(worker.join, 1.0)
            await websocket.send(json.dumps({
                "done": True,
                "interrupted": True,
                "transcript": transcript.strip(),
                "audio_ms": round(total_audio_ms),
                "metrics": metrics,
            }))
            return transcript.strip(), total_audio_ms

        await asyncio.to_thread(worker.join, 1.0)

        print(
            f"  [omni] done: {chunk_idx} audio chunks in {total_ms:.0f}ms, "
            f"audio {total_audio_ms:.0f}ms, TTFT {metrics['ttft_ms']}ms, "
            f"TPS {metrics['tps']:.2f}, TTS RTF {metrics['tts_rtf']}"
        )
        await websocket.send(json.dumps({
            "done": True,
            "transcript": transcript.strip(),
            "audio_ms": round(total_audio_ms),
            "metrics": metrics,
        }))
        return transcript.strip(), total_audio_ms

    # --- Start servers ---
    def start_http():
        httpd = http.server.HTTPServer((host, port), OmniHTTPHandler)
        httpd.serve_forever()

    http_thread = threading.Thread(target=start_http, daemon=True)
    http_thread.start()

    print(f"Starting WebSocket Omni-Live server:")
    print(f"  HTML page:  http://{host}:{port}")
    print(f"  WebSocket:  ws://{host}:{ws_port}")
    print(f"  Open http://{host}:{port} in browser")

    async def serve():
        async with websockets.serve(handle_omni, host, ws_port, origins=None, max_size=10 * 1024 * 1024):
            await asyncio.Future()

    asyncio.run(serve())


def main() -> None:
    ensure_localhost_proxy_bypass()
    ensure_ffmpeg_path()
    configure_matplotlib_for_gradio()
    args = parse_args()
    model_path = validate_model_dir(args.model_path)
    prepare_demo_assets(model_path)
    if args.validate_only:
        print(f"Model directory is ready: {model_path}")
        return

    notebook_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(notebook_dir))

    from gradio_helper import make_demo
    from minicpm_o_4_5_helper import OVMiniCPMO, _make_ov_config

    tts_device = args.tts_device or args.device
    tts_flow_device = args.tts_flow_device or tts_device
    cache_dir = args.ov_cache_dir or (Path.home() / ".cache" / "openvino" / "minicpm-o-4.5" / model_path.name)
    ov_config = _make_ov_config(args.ov_performance_hint, cache_dir)
    print(f"Loading MiniCPM-o 4.5 OpenVINO model from: {model_path}")
    print(f"OpenVINO device: {args.device}; TTS device: {tts_device}; TTS Flow device: {tts_flow_device}")
    print(f"OpenVINO config: {ov_config}")

    ov_model = OVMiniCPMO(
        model_path=str(model_path),
        device=args.device,
        tts_device=tts_device,
        ov_config=ov_config,
        tts_ov_config=ov_config,
        tts_flow_device=tts_flow_device,
    )
    if hasattr(ov_model, "as_simplex"):
        ov_model = ov_model.as_simplex()

    ref_audio_path = args.ref_audio or (model_path / "assets" / "system_ref_audio.wav")

    if args.cli_tts is not None:
        ov_model.init_tts(model_dir=args.token2wav_model_dir, hift_input_len=args.hift_input_len, n_timesteps=args.tts_timesteps)
        ref_audio = load_reference_audio(ref_audio_path)
        if ref_audio is not None:
            ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)
        output_path, transcript = generate_speech_file(
            ov_model=ov_model,
            text=args.cli_tts,
            output_path=args.output,
            ref_audio=ref_audio,
            language=args.language,
            task=args.tts_task,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Generated text: {transcript}")
        print(f"Generated audio: {output_path}")
        return

    if args.mode == "ws-tts":
        ov_model.init_tts(model_dir=args.token2wav_model_dir, hift_input_len=args.hift_input_len, n_timesteps=args.tts_timesteps)
        run_ws_tts_server(ov_model, ref_audio_path, args.language, args.host, args.port)
        return

    if args.mode == "omni":
        ov_model.init_tts(model_dir=args.token2wav_model_dir, hift_input_len=args.hift_input_len, n_timesteps=args.tts_timesteps)
        run_ws_omni_server(ov_model, ref_audio_path, args.language, args.host, args.port)
        return

    if args.mode == "tts":
        ov_model.init_tts(model_dir=args.token2wav_model_dir, hift_input_len=args.hift_input_len, n_timesteps=args.tts_timesteps)
        demo = make_tts_demo(ov_model, ref_audio_path, args.language, stream_buffer_chunks=args.stream_buffer_chunks)
    else:
        demo = make_demo(ov_model)

    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
