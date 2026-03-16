"""
MiniCPM-o 4.5 Multimodal Model with OpenVINO
完整演示程序：涵盖视觉理解、语音/音频、半双工/全双工 Omni 推理等功能。

运行前请确保已安装依赖：
    pip install "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
        "transformers==4.51.0" "openvino>=2025.1" "openvino-tokenizers>=2025.1" \
        "nncf>=2.16" "Pillow" "librosa" "soundfile" "gradio==6.0.0" \
        "accelerate" "einops" "onnxruntime" "hyperpyyaml" \
        "minicpmo-utils>=1.0.5" \
        --extra-index-url https://download.pytorch.org/whl/cpu --break-system-packages
"""

import argparse
import sys
from pathlib import Path

# 脚本所在目录，输出文件统一保存到此处
SCRIPT_DIR = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# 0. 下载工具脚本（notebook_utils.py）
# ---------------------------------------------------------------------------
import requests

notebook_utils_path = Path("notebook_utils.py")
if not notebook_utils_path.exists():
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
    )
    notebook_utils_path.write_text(r.text)


# ---------------------------------------------------------------------------
# 1. 配置路径 & 推理设备
# ---------------------------------------------------------------------------
model_id = "/home/nvme-data/AI-models/MiniCPM-o-4_5"
ov_model_path = Path("/home/nvme-data/AI-models/MiniCPM-o-4_5-OV")

# 本地 assets 目录（音频、视频等示例文件）
ASSETS_DIR = Path(model_id) / "assets"

# 默认使用 CPU；如需 GPU 请修改此处
DEVICE = "GPU"
TTS_DEVICE = "GPU"


# ---------------------------------------------------------------------------
# 2. 模型转换 & 量化（仅首次运行时执行）
# ---------------------------------------------------------------------------
def convert_model():
    from minicpm_o_4_5_helper import convert_minicpmo_model

    if not (ov_model_path / "openvino_llm_embedding_model.xml").exists():
        quantization_config = {
            "vision": {"mode": "int8_sym"},
            "llm": {"mode": "int4_sym", "group_size": 64, "ratio": 1.0},
            "text": None,
        }
        convert_minicpmo_model(
            model_id=model_id,
            output_dir=str(ov_model_path),
            quantization_config=quantization_config,
            use_stateful_apm=True,
        )
        print("✅ 全部 15 个子模型转换完成！")
    else:
        print(f"✅ 已存在转换好的模型：{ov_model_path}")


# ---------------------------------------------------------------------------
# 3. 检查本地 assets（本地模型已包含所有示例文件，无需下载）
# ---------------------------------------------------------------------------
def download_assets():
    # model_id 为本地路径时，assets 已随模型存放在 ASSETS_DIR，直接检查即可
    if Path(model_id).exists():
        files = list(ASSETS_DIR.iterdir()) if ASSETS_DIR.exists() else []
        print(f"✅ 使用本地 assets：{ASSETS_DIR}（共 {len(files)} 个文件）")
        return

    # model_id 为 HuggingFace repo ID 时，从远端下载
    from huggingface_hub import hf_hub_download, list_repo_tree

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    assets_files = []
    try:
        for item in list_repo_tree(model_id, repo_type="model"):
            if (
                item.path.startswith("assets/")
                and not item.path.startswith("assets/audio_cases")
                and not item.path.startswith("assets/token2wav")
            ):
                if "/" not in item.path[7:] or item.path.count("/") == 1:
                    assets_files.append(item.path)

        print(f"找到 {len(assets_files)} 个文件需要下载")
        for file_path in assets_files:
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=file_path,
                    local_dir=".",
                    local_dir_use_symlinks=False,
                )
                print(f"  ✅ {file_path}")
            except Exception as e:
                print(f"  ⚠️ 下载失败 {file_path}: {e}")

        print(f"\n✅ Assets 下载完成：{assets_dir.absolute()}")
    except Exception as e:
        print(f"❌ 访问仓库失败：{e}")


# ---------------------------------------------------------------------------
# 4. 初始化 OpenVINO 模型
# ---------------------------------------------------------------------------
def load_model(device=DEVICE, tts_device=TTS_DEVICE):
    from minicpm_o_4_5_helper import OVMiniCPMO

    print(f"加载模型（device={device}, tts_device={tts_device}）...")
    model = OVMiniCPMO(
        model_path=str(ov_model_path),
        device=device,
        tts_device=tts_device,
    )
    print("✅ 模型加载完成")
    return model


# ---------------------------------------------------------------------------
# 5. 视觉理解：单张图像对话
# ---------------------------------------------------------------------------
def demo_single_image(ov_model):
    from PIL import Image
    from io import BytesIO

    print("\n" + "=" * 60)
    print("【演示】单图像对话")
    print("=" * 60)

    image = None

    # 优先使用本地 assets 中的图片
    for candidate in ["fossil.png", "highway.png"]:
        local_img = ASSETS_DIR / candidate
        if local_img.exists():
            image = Image.open(str(local_img)).convert("RGB")
            print(f"✅ 使用本地图片：{local_img}")
            break

    # 本地无图片时尝试从网络下载
    if image is None:
        image_url = (
            "https://github.com/openvinotoolkit/openvino_notebooks/assets/"
            "29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11"
        )
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            print(f"✅ 图片下载成功")
        except Exception as e:
            print(f"❌ 图片获取失败：{e}")
            return None, None

    image.thumbnail((512, 512))

    question = "What is in the image?"
    msgs = [{"role": "user", "content": [image, question]}]

    answer = ov_model.chat(msgs=msgs, use_tts_template=False)
    print(f"Q: {question}")
    print(f"A: {answer}")
    return image, answer


# ---------------------------------------------------------------------------
# 6. 视觉理解：多轮对话
# ---------------------------------------------------------------------------
def demo_multi_turn(ov_model, image, first_answer):
    print("\n" + "=" * 60)
    print("【演示】多轮对话")
    print("=" * 60)

    ov_model.llm._ov_language.reset_state()
    ov_model.llm._past_length = 0

    msgs = [
        {"role": "user", "content": [image, "What is in this image?"]},
        {"role": "assistant", "content": [first_answer]},
        {"role": "user", "content": ["What season do you think this picture was taken in? Why?"]},
    ]

    answer2 = ov_model.chat(msgs=msgs, use_tts_template=False, enable_thinking=False)
    print(f"A: {answer2}")
    return answer2


# ---------------------------------------------------------------------------
# 7. 语音/音频理解：ASR（自动语音识别）
# ---------------------------------------------------------------------------
def demo_asr(ov_model):
    import librosa
    import soundfile as sf

    print("\n" + "=" * 60)
    print("【演示】音频理解 - ASR")
    print("=" * 60)

    audio_path = ASSETS_DIR / "system_ref_audio.wav"
    if not Path(audio_path).exists():
        print(f"⚠️ 找不到音频文件 {audio_path}，跳过 ASR 演示")
        return None

    audio_input, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    print(f"✅ 音频加载成功：{len(audio_input)/16000:.1f}s @ 16kHz")

    ov_model.llm._ov_language.reset_state()
    ov_model.llm._past_length = 0

    task_prompt = "Please listen to the audio snippet carefully and transcribe the content."
    msgs = [{"role": "user", "content": [task_prompt, audio_input]}]

    asr_result = ov_model.chat(
        msgs=msgs,
        do_sample=True,
        max_new_tokens=512,
        use_tts_template=True,
        temperature=0.3,
    )
    print(f"ASR 结果：{asr_result}")
    return audio_input


# ---------------------------------------------------------------------------
# 8. 音频理解：说话人分析 & 场景标注
# ---------------------------------------------------------------------------
def demo_more_audio(ov_model, audio_input):
    print("\n" + "=" * 60)
    print("【演示】音频理解 - 说话人分析 & 场景标注")
    print("=" * 60)

    # 说话人分析
    ov_model.llm._ov_language.reset_state()
    ov_model.llm._past_length = 0
    speaker_prompt = "Based on the speaker's content, speculate on their gender, condition, age range, and health status."
    msgs = [{"role": "user", "content": [speaker_prompt, audio_input]}]
    speaker_result = ov_model.chat(
        msgs=msgs, do_sample=True, max_new_tokens=512, use_tts_template=True, temperature=0.3
    )
    print(f"说话人分析：{speaker_result}")

    # 场景标注
    ov_model.llm._ov_language.reset_state()
    ov_model.llm._past_length = 0
    scene_prompt = "Utilize one keyword to convey the audio's content or the associated scene."
    msgs = [{"role": "user", "content": [scene_prompt, audio_input]}]
    scene_result = ov_model.chat(
        msgs=msgs, do_sample=True, max_new_tokens=64, use_tts_template=True, temperature=0.3
    )
    print(f"场景标注：{scene_result}")


# ---------------------------------------------------------------------------
# 9. 半双工 Omni 模式：Chat 推理
# ---------------------------------------------------------------------------
def demo_halfduplex_omni_chat(ov_model):
    from minicpmo.utils import get_video_frame_audio_segments

    print("\n" + "=" * 60)
    print("【演示】半双工 Omni 模式 - Chat 推理")
    print("=" * 60)

    ov_model = ov_model.as_duplex()
    ov_model.init_tts()

    video_path = ASSETS_DIR / "Skiing.mp4"
    if not Path(video_path).exists():
        print(f"⚠️ 找不到视频文件 {video_path}，跳过此演示")
        return ov_model

    ref_audio_path = ASSETS_DIR / "HT_ref_audio.wav"
    sys_msg = ov_model.get_sys_prompt(ref_audio=str(ref_audio_path), mode="omni", language="en")

    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        video_path, stack_frames=1
    )
    omni_contents = []
    for i in range(len(video_frames)):
        omni_contents.append(video_frames[i])
        omni_contents.append(audio_segments[i])
        if stacked_frames is not None and stacked_frames[i] is not None:
            omni_contents.append(stacked_frames[i])

    msg = {"role": "user", "content": omni_contents}
    msgs = [sys_msg, msg]

    output_audio_path = str(SCRIPT_DIR / "output.wav")
    res = ov_model.chat(
        msgs=msgs,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.7,
        use_tts_template=True,
        enable_thinking=False,
        omni_mode=True,
        generate_audio=True,
        output_audio_path=output_audio_path,
        max_slice_nums=1,
    )
    print(f"响应文本：{res}")
    print(f"音频已保存至：{output_audio_path}")
    return ov_model


# ---------------------------------------------------------------------------
# 10. 半双工 Omni 模式：流式推理
# ---------------------------------------------------------------------------
def demo_halfduplex_omni_streaming(ov_model):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from minicpmo.utils import get_video_frame_audio_segments

    print("\n" + "=" * 60)
    print("【演示】半双工 Omni 模式 - 流式推理")
    print("=" * 60)

    ov_model = ov_model.as_duplex()
    ov_model.init_tts()
    ov_model.reset_session()

    video_path = ASSETS_DIR / "Skiing.mp4"
    ref_audio_path = ASSETS_DIR / "HT_ref_audio.wav"

    if not Path(video_path).exists() or not Path(ref_audio_path).exists():
        print("⚠️ 缺少视频或参考音频文件，跳过此演示")
        return ov_model

    ref_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)
    ov_model.init_token2wav_cache(ref_audio)

    session_id = "demo"
    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        str(video_path), stack_frames=1
    )

    omni_contents = []
    for i in range(len(video_frames)):
        omni_contents.append(video_frames[i])
        omni_contents.append(audio_segments[i])
        if stacked_frames is not None and stacked_frames[i] is not None:
            omni_contents.append(stacked_frames[i])

    # Step 1: 预填充系统提示
    sys_msg = ov_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
    ov_model.streaming_prefill(session_id=session_id, msgs=[sys_msg])

    # Step 2: 逐块预填充
    audio_indices = [i for i, c in enumerate(omni_contents) if isinstance(c, np.ndarray)]
    last_audio_idx = audio_indices[-1] if audio_indices else -1

    for idx, content in enumerate(omni_contents):
        is_last_audio_chunk = idx == last_audio_idx
        msgs = [{"role": "user", "content": [content]}]
        ov_model.streaming_prefill(
            session_id=session_id,
            msgs=msgs,
            omni_mode=True,
            is_last_chunk=is_last_audio_chunk,
        )

    # Step 3: 流式生成
    output_audio_path = str(SCRIPT_DIR / "output.wav")
    iter_gen = ov_model.streaming_generate(
        session_id=session_id,
        generate_audio=True,
        use_tts_template=True,
        enable_thinking=False,
        do_sample=True,
    )

    audios = []
    text = ""
    for wav_chunk, text_chunk in iter_gen:
        audios.append(wav_chunk)
        text += text_chunk

    generated_waveform = torch.cat(audios, dim=-1)[0]
    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)
    print(f"文本：{text}")
    print(f"音频已保存至：{output_audio_path}")
    return ov_model


# ---------------------------------------------------------------------------
# 11. 半双工实时语音对话
# ---------------------------------------------------------------------------
def demo_realtime_speech(ov_model):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    print("\n" + "=" * 60)
    print("【演示】半双工实时语音对话")
    print("=" * 60)

    ov_model = ov_model.as_duplex()

    ref_audio_path = ASSETS_DIR / "system_ref_audio.wav"
    if not Path(ref_audio_path).exists():
        print(f"⚠️ 找不到参考音频 {ref_audio_path}，跳过此演示")
        return ov_model

    ref_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)

    sys_msg = {
        "role": "system",
        "content": [
            "Clone the voice in the provided audio prompt.",
            ref_audio,
            "Please assist users while maintaining this voice style. "
            "Please answer the user's questions seriously and in a high quality. "
            "Please chat with the user in a highly human-like and oral style in English. "
            "You are a helpful assistant developed by ModelBest: MiniCPM-Omni",
        ],
    }

    ov_model.reset_session(reset_token2wav_cache=True)
    ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

    session_id = "demo"
    ov_model.streaming_prefill(
        session_id=session_id,
        msgs=[sys_msg],
        omni_mode=False,
        is_last_chunk=True,
    )

    # 模拟实时输入：将参考音频按 1s 分块发送
    user_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)
    IN_SAMPLE_RATE = 16000
    CHUNK_SAMPLES = IN_SAMPLE_RATE
    MIN_AUDIO_SAMPLES = 16000

    total_samples = len(user_audio)
    num_chunks = (total_samples + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

    for chunk_idx in range(num_chunks):
        start = chunk_idx * CHUNK_SAMPLES
        end = min((chunk_idx + 1) * CHUNK_SAMPLES, total_samples)
        chunk_audio = user_audio[start:end]

        is_last_chunk = chunk_idx == num_chunks - 1
        if is_last_chunk and len(chunk_audio) < MIN_AUDIO_SAMPLES:
            chunk_audio = np.concatenate(
                [chunk_audio, np.zeros(MIN_AUDIO_SAMPLES - len(chunk_audio), dtype=chunk_audio.dtype)]
            )

        user_msg = {"role": "user", "content": [chunk_audio]}
        ov_model.streaming_prefill(
            session_id=session_id,
            msgs=[user_msg],
            omni_mode=False,
            is_last_chunk=is_last_chunk,
        )

    iter_gen = ov_model.streaming_generate(
        session_id=session_id,
        generate_audio=True,
        use_tts_template=True,
        enable_thinking=False,
        do_sample=True,
        max_new_tokens=512,
        length_penalty=1.1,
    )

    audios = []
    text = ""
    for wav_chunk, text_chunk in iter_gen:
        audios.append(wav_chunk)
        text += text_chunk

    output_audio_path = str(SCRIPT_DIR / "output_realtime.wav")
    generated_waveform = torch.cat(audios, dim=-1)[0]
    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)
    print(f"文本：{text}")
    print(f"音频已保存至：{output_audio_path}")
    return ov_model


# ---------------------------------------------------------------------------
# 12. 全双工 Omni 模式
# ---------------------------------------------------------------------------
def demo_duplex_omni(ov_model):
    import librosa
    import torch
    from minicpmo.utils import generate_duplex_video, get_video_frame_audio_segments

    print("\n" + "=" * 60)
    print("【演示】全双工 Omni 模式")
    print("=" * 60)

    ov_model = ov_model.as_duplex() if hasattr(ov_model, "as_duplex") else ov_model

    cn_video_path = ASSETS_DIR / "omni_duplex1.mp4"
    ref_audio_path = ASSETS_DIR / "HT_ref_audio.wav"
    en_video_url = "https://github.com/user-attachments/assets/f6ab8ed5-5829-4ce2-9feb-3a8bf7374dbb"
    en_video_path = ASSETS_DIR / "omni_duplex_en.mp4"

    try:
        resp = requests.get(en_video_url, timeout=30)
        resp.raise_for_status()
        if not resp.content:
            raise ValueError("下载内容为空")
        with open(en_video_path, "wb") as f:
            f.write(resp.content)
        video_path = en_video_path
        print(f"✅ 英文视频下载完成")
    except Exception as e:
        if Path(cn_video_path).exists():
            video_path = cn_video_path
            print(f"⚠️ 下载失败，使用备用视频 {cn_video_path}：{e}")
        else:
            print(f"⚠️ 视频文件不可用，跳过全双工演示：{e}")
            return ov_model

    if not Path(ref_audio_path).exists():
        print(f"⚠️ 找不到参考音频 {ref_audio_path}，跳过此演示")
        return ov_model

    ref_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)
    ov_model.reset_session(reset_token2wav_cache=True)

    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        str(video_path), stack_frames=1, use_ffmpeg=True, adjust_audio_length=True
    )

    ov_model.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=str(ref_audio_path),
    )

    results_log = []
    timed_output_audio = []

    for chunk_idx in range(len(audio_segments)):
        audio_chunk = audio_segments[chunk_idx] if chunk_idx < len(audio_segments) else None
        frame = video_frames[chunk_idx] if chunk_idx < len(video_frames) else None
        frame_list = []
        if frame is not None:
            frame_list.append(frame)
            if (
                stacked_frames is not None
                and chunk_idx < len(stacked_frames)
                and stacked_frames[chunk_idx] is not None
            ):
                frame_list.append(stacked_frames[chunk_idx])

        ov_model.streaming_prefill(
            audio_waveform=audio_chunk,
            frame_list=frame_list,
            max_slice_nums=1,
            batch_vision_feed=False,
        )

        result = ov_model.streaming_generate(
            prompt_wav_path=str(ref_audio_path),
            max_new_speak_tokens_per_chunk=20,
            decode_mode="sampling",
        )

        if result["audio_waveform"] is not None:
            timed_output_audio.append((chunk_idx, result["audio_waveform"]))

        chunk_result = {
            "chunk_idx": chunk_idx,
            "is_listen": result["is_listen"],
            "text": result["text"],
            "end_of_turn": result["end_of_turn"],
            "current_time": result["current_time"],
            "audio_length": len(result["audio_waveform"]) if result["audio_waveform"] is not None else 0,
        }
        results_log.append(chunk_result)
        print("监听中..." if result["is_listen"] else f"输出> {result['text']}")

    generate_duplex_video(
        video_path=video_path,
        output_video_path=str(SCRIPT_DIR / "duplex_output.mp4"),
        results_log=results_log,
        timed_output_audio=timed_output_audio,
        output_sample_rate=24000,
    )
    print("✅ 全双工视频已保存至 duplex_output.mp4")
    return ov_model


# ---------------------------------------------------------------------------
# 13. 交互式 Gradio Demo
# ---------------------------------------------------------------------------
def launch_gradio_demo(ov_model):
    print("\n" + "=" * 60)
    print("【演示】启动 Gradio 交互式界面")
    print("=" * 60)

    from gradio_helper import make_demo

    # 切换到 duplex 模式并初始化 TTS，才能支持语音回复
    if hasattr(ov_model, "as_duplex"):
        ov_model = ov_model.as_duplex()
    ov_model.init_tts()

    demo = make_demo(ov_model, assets_dir=ASSETS_DIR)
    demo.launch(debug=False, server_name="0.0.0.0", server_port=7860, height=800)


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 OpenVINO 完整演示")
    parser.add_argument(
        "--convert", action="store_true", help="执行模型转换（仅首次需要）"
    )
    parser.add_argument(
        "--download-assets", action="store_true", help="从 HuggingFace 下载示例 assets"
    )
    parser.add_argument(
        "--device", default="GPU", help="主推理设备（默认 GPU）"
    )
    parser.add_argument(
        "--tts-device", default="GPU", help="TTS 推理设备（默认 GPU）"
    )
    parser.add_argument(
        "--demo",
        choices=[
            "image",
            "multi_turn",
            "asr",
            "audio",
            "halfduplex_chat",
            "halfduplex_stream",
            "realtime_speech",
            "duplex",
            "gradio",
            "all",
        ],
        default="image",
        help="选择要运行的演示（默认 image）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.convert:
        convert_model()

    if args.download_assets:
        download_assets()

    ov_model = load_model(device=args.device, tts_device=args.tts_device)

    demo = args.demo

    image, first_answer = None, None

    if demo in ("image", "multi_turn", "all"):
        image, first_answer = demo_single_image(ov_model)

    if demo in ("multi_turn", "all") and image is not None:
        demo_multi_turn(ov_model, image, first_answer)

    audio_input = None
    if demo in ("asr", "audio", "all"):
        audio_input = demo_asr(ov_model)

    if demo in ("audio", "all") and audio_input is not None:
        demo_more_audio(ov_model, audio_input)

    if demo in ("halfduplex_chat", "all"):
        ov_model = demo_halfduplex_omni_chat(ov_model)

    if demo in ("halfduplex_stream", "all"):
        ov_model = demo_halfduplex_omni_streaming(ov_model)

    if demo in ("realtime_speech", "all"):
        ov_model = demo_realtime_speech(ov_model)

    if demo in ("duplex", "all"):
        ov_model = demo_duplex_omni(ov_model)

    if demo == "gradio":
        launch_gradio_demo(ov_model)


if __name__ == "__main__":
    main()
