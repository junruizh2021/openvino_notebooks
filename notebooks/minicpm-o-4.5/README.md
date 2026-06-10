# MiniCPM-o 4.5 Multimodal Model with OpenVINO

<p align="center">
    <img src="https://raw.githubusercontent.com/OpenBMB/MiniCPM-o/main/assets/minicpm-o-45-framework.png" width="100%"/>
</p>

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is the latest and most capable model in the MiniCPM-o series — an end-to-end omnimodal model with **9B parameters** built on **SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B**. It achieves Gemini 2.5 Flash level performance on vision-language benchmarks with only 9B parameters.

## Key Features

- **Leading Visual Capability** — 77.6 on OpenCompass, surpassing GPT-4o and Gemini 2.0 Pro
- **Strong Speech Capability** — Bilingual (EN/ZH) real-time speech conversation with configurable voice cloning
- **Full-Duplex Live Streaming** — See, listen, and speak simultaneously with real-time video + audio
- **Proactive Interaction** — Initiates reminders/comments based on live scene understanding
- **Strong OCR** — State-of-the-art end-to-end English document parsing

## Architecture (16 Sub-models)

The model comprises 16 interconnected sub-models converted to OpenVINO IR format:

| Sub-model | Role | Quantization |
|-----------|------|:------------:|
| **LLM Embedding** | Token embeddings for Qwen3-8B backbone | — |
| **LLM Language Model** | Main language model with stateful KV cache | INT4 |
| **Vision Model** | SigLip2 vision encoder for image understanding | INT8 |
| **Resampler** | Projects vision features to LLM hidden space | — |
| **Audio Encoder** | Whisper-medium encoder for speech/audio understanding | — |
| **Audio Projection** | Projects audio features to LLM hidden space | — |
| **TTS Text Embedding** | Token embeddings for TTS LLaMA decoder | — |
| **TTS Language Model** | LLaMA decoder for speech token generation | — |
| **TTS Projector SPK** | Speaker embedding projector | — |
| **TTS Projector Semantic** | Semantic feature projector for TTS conditioning | — |
| **TTS Code Embedding** | Audio code token embeddings | — |
| **TTS Code Head** | Audio code prediction head | — |
| **Flow Embeddings** | CosyVoice2 flow-matching encoder + speaker projection | — |
| **Flow Encoder Chunk** | Streaming conformer encoder with KV cache | — |
| **Flow Estimator Chunk** | Unified DiT for flow-matching (streaming & non-streaming) | — |
| **HiFT** | Neural vocoder for mel-to-waveform synthesis | — |

> **Note:** The Flow Estimator Chunk model serves both streaming and non-streaming inference. In non-streaming mode it runs with empty KV caches (bit-identical to the legacy full estimator), while in streaming mode the KV caches enable temporally coherent cross-chunk mel generation — aligned with the original CosyVoice2 `flow.inference_chunk()` design. This unified approach saves ~220MB of memory.

## Notebook Contents

The notebook demonstrates:

1. **Prerequisites** — Install dependencies
2. **Convert & Quantize Model** — Export all 16 sub-models to OpenVINO IR with INT4/INT8 weight compression
3. **Select Inference Device** — Choose CPU, GPU, or NPU for different model components
4. **Run Inference** — Image understanding, audio understanding, omni-modal chat
5. **Interactive Demo** — Gradio-based multimodal chatbot

In this demonstration, you'll create an interactive chatbot that can answer questions about the provided image's content.

The image below illustrates example of input prompt and model answer.
![example.png](https://github.com/user-attachments/assets/906c5b2d-aa90-4d46-b417-9421b2061da2)

## Installation instructions
This is a self-contained example that relies solely on its own code.
We recommend running the notebook in a virtual environment. You only need a Jupyter server to start.
For details, please refer to [Installation Guide](../../README.md).

## Run The Web Demo

If the converted OpenVINO model is already available locally, start the Gradio
web demo directly:

```bash
cd /home/junruizh/openvino_notebooks/notebooks/minicpm-o-4.5
python omni_live_server.py \
    --model-path ~/junrui/MiniCPM-o-4_5-OV \
    --device GPU \
    --ov-performance-hint LATENCY \
    --ov-cache-dir $HOME/.cache/openvino/minicpm-o-4.5/MiniCPM-o-4_5-OV \
    --host 0.0.0.0 \
    --port 7860
```

Then open `http://127.0.0.1:7860`. The page supports text chat, image upload,
audio upload, and microphone input.

To test text input with speech output first:

```bash
cd /home/junruizh/openvino_notebooks/notebooks/minicpm-o-4.5
python omni_live_server.py \
    --cli-tts "请用自然的中文语音介绍一下 OpenVINO 是什么。" \
    --output minicpmo_tts_output.wav \
    --model-path /home/nvme-data/AI-models/MiniCPM-o-4_5-OV \
    --device GPU \
    --tts-device GPU \
    --tts-flow-device GPU \
    --tts-timesteps 3 \
    --hift-input-len 64 \
    --token2wav-model-dir ./token2wav_chunk_ov \
    --ov-performance-hint LATENCY \
    --ov-cache-dir $HOME/.cache/openvino/minicpm-o-4.5/MiniCPM-o-4_5-OV
```

Or run the TTS page:

```bash
cd /home/junruizh/openvino_notebooks/notebooks/minicpm-o-4.5
python omni_live_server.py \
    --mode tts \
    --model-path /home/nvme-data/AI-models/MiniCPM-o-4_5-OV \
    --device GPU \
    --tts-device GPU \
    --tts-flow-device GPU \
    --tts-timesteps 3 \
    --hift-input-len 64 \
    --token2wav-model-dir ./token2wav_chunk_ov \
    --ov-performance-hint LATENCY \
    --ov-cache-dir $HOME/.cache/openvino/minicpm-o-4.5/MiniCPM-o-4_5-OV \
    --host 0.0.0.0 \
    --port 7860
```

The TTS page provides two actions:

- `Generate Speech`: writes the final WAV file after the full response is generated.
- `Stream Speech`: updates the preview audio and transcript incrementally as audio chunks are produced.

## TTS Performance Tuning

For real-time conversation on Intel PTL iGPU, tune the TTS path separately from
the main chat model. The server exposes the main latency controls:

- `--tts-device GPU`: runs TTS language model, Flow estimator, HiFT, and speaker model on GPU.
- `--tts-flow-device GPU`: runs Flow embedding and streaming encoder on GPU instead of the previous CPU-only path.
- `--tts-timesteps N`: controls Flow matching denoising steps. Lower values are faster; default is `3`; test `4`, `6`, `8`, and `10` for quality/RTF tradeoff.
- `--hift-input-len 64`: keeps HiFT static for the normal 25-token streaming chunk and avoids GPU dynamic-shape overhead.
- `--token2wav-model-dir ./token2wav_chunk_ov`: loads chunk Flow + cached HiFT models from a separately exported token2wav directory.
- `--ov-performance-hint LATENCY`: asks OpenVINO to optimize compiled models for low latency.
- `--ov-cache-dir <dir>`: enables OpenVINO compiled-model caching so repeated launches avoid GPU compile overhead.
- `--stream-buffer-chunks N`: accumulates N model audio chunks before yielding to Gradio streaming. Higher values (2-3) reduce ffmpeg/HTTP/browser overhead and smooth playback, but increase first-chunk latency. Default `1` (no accumulation).

Recommended PTL iGPU sweep:

```bash
for steps in 3 4 6 8 10; do
  python omni_live_server.py \
      --cli-tts "请用自然的中文语音介绍一下 OpenVINO 是什么。" \
      --output "minicpmo_tts_steps_${steps}.wav" \
      --model-path /home/nvme-data/AI-models/MiniCPM-o-4_5-OV \
      --device GPU \
      --tts-device GPU \
      --tts-flow-device GPU \
      --tts-timesteps "${steps}" \
      --hift-input-len 64 \
      --token2wav-model-dir ./token2wav_chunk_ov \
      --ov-performance-hint LATENCY \
      --ov-cache-dir $HOME/.cache/openvino/minicpm-o-4.5/MiniCPM-o-4_5-OV
done
```

If GPU RTF is still above 1, compare a mixed placement because short dynamic
chunks can be CPU-bound by scheduling and tensor copies:

```bash
python omni_live_server.py \
    --cli-tts "请用自然的中文语音介绍一下 OpenVINO 是什么。" \
    --output minicpmo_tts_mixed.wav \
    --model-path /home/nvme-data/AI-models/MiniCPM-o-4_5-OV \
    --device CPU \
    --tts-device GPU \
    --tts-flow-device CPU \
    --tts-timesteps 3 \
    --hift-input-len 64 \
    --token2wav-model-dir ./token2wav_chunk_ov \
    --ov-performance-hint LATENCY
```

TTS LLM weight compression is also supported during conversion by passing a
`quantization_config` with a `tts` entry, for example `{"tts": {"mode":
"int8_sym"}}`. Keep this optional and validate audio quality, because it changes
the autoregressive speech-token generator.

## WebSocket Low-Latency TTS (Recommended)

For the smoothest streaming playback experience, use `--mode ws-tts` which
bypasses Gradio/ffmpeg entirely and sends raw PCM over WebSocket to an
AudioWorklet-based player:

```bash
cd /home/junruizh/openvino_notebooks/notebooks/minicpm-o-4.5
python omni_live_server.py \
    --mode ws-tts \
    --model-path ~/junrui/MiniCPM-o-4_5-OV \
    --device GPU \
    --tts-device GPU \
    --tts-flow-device GPU \
    --tts-timesteps 3 \
    --hift-input-len 64 \
    --token2wav-model-dir ./token2wav_chunk_ov \
    --ov-performance-hint LATENCY \
    --ov-cache-dir $HOME/.cache/openvino/minicpm-o-4.5/MiniCPM-o-4_5-OV \
    --host 0.0.0.0 \
    --port 7860
```

Then open `http://<host>:7860`. The player features:

- **Zero ffmpeg overhead** — raw float32 PCM sent as WebSocket binary frames
- **AudioWorklet playback** — browser-native low-latency audio rendering
- **Configurable jitter buffer** — default 300ms pre-buffer before playback starts
- **Real-time diagnostics** — buffer level, chunk count, first-chunk latency

Requires `fastapi` and `uvicorn` (`pip install fastapi uvicorn`).

⚠️ **EXPERIMENTAL NOTEBOOK**

This notebook demonstrates a model that has not been fully validated with OpenVINO. It may be fully supported and validated in the future.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/minicpm-o-4.5/README.md" />
