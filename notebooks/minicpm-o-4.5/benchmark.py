"""
MiniCPM-o 4.5 性能基准测试
测试场景：
  1. 纯文本输入
  2. 文本 + 图片输入
  3. 文本 + 音频输入
  4. 文本输入 + 音频输出（TTS）

每个场景测试前先用 "你好" 进行一次预热。

用法示例
--------
# 运行全部场景（默认 3 次/场景，GPU 推理）
python benchmark.py

# 只测纯文本场景
python benchmark.py --scenarios text

# 只测文本+图片场景
python benchmark.py --scenarios image

# 只测文本+音频输入场景
python benchmark.py --scenarios audio

# 只测文本+音频输出（TTS）场景
python benchmark.py --scenarios tts

# TTS 场景并实时流式播放音频
python benchmark.py --scenarios tts --play

# 指定多个场景
python benchmark.py --scenarios text image

# 指定测试次数（每场景 5 次）
python benchmark.py --runs 5

# 使用 CPU 推理
python benchmark.py --device CPU --tts-device CPU

# 组合参数：只测 TTS，运行 5 次，CPU 推理，边生成边播放
python benchmark.py --scenarios tts --runs 5 --device CPU --tts-device CPU --play
"""

import argparse
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径配置（与 minicpm_o_4_5_demo.py 保持一致）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
model_id = "/home/nvme-data/AI-models/MiniCPM-o-4_5"
ov_model_path = Path("/home/nvme-data/AI-models/MiniCPM-o-4_5-OV")
ASSETS_DIR = Path(model_id) / "assets"

DEFAULT_DEVICE = "GPU"
DEFAULT_TTS_DEVICE = "GPU"
DEFAULT_RUNS = 3


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def reset_model_state(ov_model):
    """重置 LLM KV-cache 状态，确保每次推理独立。"""
    try:
        ov_model.llm._ov_language.reset_state()
        ov_model.llm._past_length = 0
    except Exception:
        pass


def count_tokens(ov_model, text: str) -> int:
    """统计文本 token 数。"""
    try:
        return len(ov_model.tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text.split())


def measure_chat_stream(ov_model, msgs, chat_kwargs: dict) -> dict:
    """
    使用流式模式测量推理性能。
    返回字段：
      - ttft_s     : 首 token 时延（秒）
      - total_s    : 总生成时间（秒）
      - answer     : 生成文本
      - num_tokens : 输出 token 数
      - tps        : 吞吐量（tokens/s，基于总时间）
    """
    kw = dict(chat_kwargs)
    kw["stream"] = True

    reset_model_state(ov_model)
    t0 = time.perf_counter()

    streamer = ov_model.chat(msgs=msgs, **kw)

    ttft_s = None
    answer = ""
    for chunk in streamer:
        if chunk:
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            answer += chunk

    total_s = time.perf_counter() - t0
    if ttft_s is None:
        ttft_s = total_s

    num_tokens = count_tokens(ov_model, answer)
    tps = num_tokens / total_s if total_s > 0 else 0.0

    return {
        "ttft_s": ttft_s,
        "total_s": total_s,
        "answer": answer,
        "num_tokens": num_tokens,
        "tps": tps,
    }


def measure_chat_non_stream(ov_model, msgs, chat_kwargs: dict) -> dict:
    """
    使用非流式模式测量推理性能（用于含音频输出的场景，stream 不支持同步音频生成）。
    返回字段与 measure_chat_stream 一致，ttft_s 填充 None。
    """
    kw = dict(chat_kwargs)
    kw.pop("stream", None)

    reset_model_state(ov_model)
    t0 = time.perf_counter()
    answer = ov_model.chat(msgs=msgs, **kw)
    total_s = time.perf_counter() - t0

    if answer is None:
        answer = ""

    num_tokens = count_tokens(ov_model, answer)
    tps = num_tokens / total_s if total_s > 0 else 0.0

    return {
        "ttft_s": None,
        "total_s": total_s,
        "answer": answer,
        "num_tokens": num_tokens,
        "tps": tps,
    }


def run_benchmark(name: str, ov_model, warmup_fn, bench_fn, num_runs: int):
    """
    通用 benchmark 执行器：先预热，再多次测量，打印统计结果。

    Args:
        name       : 场景名称
        ov_model   : 已加载的模型
        warmup_fn  : 预热函数，签名 () -> None
        bench_fn   : 单次测量函数，签名 () -> dict（字段见 measure_* 函数）
        num_runs   : 测量次数
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  场景：{name}")
    print(sep)

    # ---------- 预热 ----------
    print("  [预热] 输入：你好 ...")
    try:
        warmup_fn()
    except Exception as e:
        print(f"  [预热] 警告：{e}")
    print("  [预热] 完成\n")

    # ---------- 测量 ----------
    results = []
    for i in range(num_runs):
        print(f"  [Run {i + 1}/{num_runs}] 推理中...", end="", flush=True)
        try:
            r = bench_fn()
            results.append(r)
            ttft_str = f"{r['ttft_s'] * 1000:.1f} ms" if r["ttft_s"] is not None else "N/A"
            print(
                f" 完成 | 总耗时={r['total_s']:.2f}s  TTFT={ttft_str}"
                f"  tokens={r['num_tokens']}  TPS={r['tps']:.1f}"
            )
        except Exception as e:
            print(f" 错误：{e}")

    if not results:
        print("  所有 run 均失败，跳过统计。")
        return

    # ---------- 统计 ----------
    total_ss = [r["total_s"] for r in results]
    tps_list = [r["tps"] for r in results]
    ttft_list = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    tokens_list = [r["num_tokens"] for r in results]

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    def mn(lst):
        return min(lst) if lst else 0.0

    def mx(lst):
        return max(lst) if lst else 0.0

    print(f"\n  ── 统计（{len(results)} 次）──")
    print(f"  总耗时   avg={avg(total_ss):.2f}s  min={mn(total_ss):.2f}s  max={mx(total_ss):.2f}s")
    print(f"  TPS      avg={avg(tps_list):.1f}  min={mn(tps_list):.1f}  max={mx(tps_list):.1f}  (tokens/s)")
    if ttft_list:
        print(
            f"  TTFT     avg={avg(ttft_list) * 1000:.1f}ms"
            f"  min={mn(ttft_list) * 1000:.1f}ms"
            f"  max={mx(ttft_list) * 1000:.1f}ms"
        )
    else:
        print("  TTFT     N/A（非流式模式）")
    print(f"  输出tokens avg={avg(tokens_list):.0f}  min={mn(tokens_list)}  max={mx(tokens_list)}")

    # 打印最后一次的回答（截断显示）
    last_answer = results[-1]["answer"]
    preview = last_answer[:120].replace("\n", " ")
    if len(last_answer) > 120:
        preview += "..."
    print(f"\n  最后一次回答预览：{preview}")
    print(sep)

    return results


# ---------------------------------------------------------------------------
# 场景 1：纯文本输入
# ---------------------------------------------------------------------------
def benchmark_text_only(ov_model, num_runs: int):
    question = "介绍一下你自己吧"
    chat_kwargs = {
        "use_tts_template": False,
        "max_new_tokens": 256,
        "do_sample": False,
    }

    def warmup():
        reset_model_state(ov_model)
        msgs = [{"role": "user", "content": "你好"}]
        ov_model.chat(msgs=msgs, use_tts_template=False, max_new_tokens=64, do_sample=False)

    def bench():
        msgs = [{"role": "user", "content": question}]
        return measure_chat_stream(ov_model, msgs, chat_kwargs)

    run_benchmark("文本输入", ov_model, warmup, bench, num_runs)


# ---------------------------------------------------------------------------
# 场景 2：文本 + 图片输入
# ---------------------------------------------------------------------------
def benchmark_text_image(ov_model, num_runs: int):
    from PIL import Image

    # 寻找可用图片
    image = None
    for candidate in ["fossil.png", "highway.png"]:
        local_img = ASSETS_DIR / candidate
        if local_img.exists():
            image = Image.open(str(local_img)).convert("RGB")
            image.thumbnail((512, 512))
            print(f"  使用图片：{local_img}")
            break

    if image is None:
        print("  ⚠️ 未找到本地图片文件，跳过此场景。")
        print(f"  请确认 {ASSETS_DIR} 下存在 fossil.png 或 highway.png")
        return

    question = "介绍图片中的内容"
    chat_kwargs = {
        "use_tts_template": False,
        "max_new_tokens": 256,
        "do_sample": False,
    }

    def warmup():
        reset_model_state(ov_model)
        msgs = [{"role": "user", "content": "你好"}]
        ov_model.chat(msgs=msgs, use_tts_template=False, max_new_tokens=64, do_sample=False)

    def bench():
        msgs = [{"role": "user", "content": [image, question]}]
        return measure_chat_stream(ov_model, msgs, chat_kwargs)

    run_benchmark("文本 + 图片输入", ov_model, warmup, bench, num_runs)


# ---------------------------------------------------------------------------
# 场景 3：文本 + 音频输入
# ---------------------------------------------------------------------------
def benchmark_text_audio(ov_model, num_runs: int):
    import librosa

    audio_path = ASSETS_DIR / "system_ref_audio.wav"
    if not audio_path.exists():
        print(f"  ⚠️ 未找到音频文件 {audio_path}，跳过此场景。")
        return

    audio_input, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    duration = len(audio_input) / 16000
    print(f"  使用音频：{audio_path}（时长 {duration:.1f}s）")

    task_prompt = "介绍音频中的内容"
    chat_kwargs = {
        "use_tts_template": True,
        "max_new_tokens": 256,
        "do_sample": False,
    }

    def warmup():
        reset_model_state(ov_model)
        msgs = [{"role": "user", "content": "你好"}]
        ov_model.chat(msgs=msgs, use_tts_template=False, max_new_tokens=64, do_sample=False)

    def bench():
        msgs = [{"role": "user", "content": [task_prompt, audio_input]}]
        return measure_chat_stream(ov_model, msgs, chat_kwargs)

    run_benchmark("文本 + 音频输入", ov_model, warmup, bench, num_runs)


# ---------------------------------------------------------------------------
# 场景 4：文本输入 + 音频输出（TTS）
# 使用 streaming_prefill + streaming_generate API（需要参考音频来克隆声音）
# ---------------------------------------------------------------------------
def benchmark_text_audio_output(ov_model, num_runs: int, play_audio: bool = False):
    import librosa
    import soundfile as sf
    import torch

    # 参考音频（用于 TTS 声音克隆）
    ref_audio_path = ASSETS_DIR / "system_ref_audio.wav"
    if not ref_audio_path.exists():
        print(f"  ⚠️ 未找到参考音频 {ref_audio_path}，跳过此场景。")
        return

    ref_audio, _ = librosa.load(str(ref_audio_path), sr=16000, mono=True)
    print(f"  使用参考音频：{ref_audio_path}")

    question = "介绍一下你自己吧"
    output_audio_path = str(SCRIPT_DIR / "benchmark_tts_output.wav")
    session_id = "benchmark_tts"

    def _run_tts(text: str, save_path: str | None, max_new_tokens: int = 256) -> dict:
        """执行一次 TTS 推理，返回性能指标。"""
        ov_model.reset_session(reset_token2wav_cache=True)
        ov_model.init_token2wav_cache(prompt_speech_16k=ref_audio)

        user_msg = {"role": "user", "content": text}

        t0 = time.perf_counter()
        ov_model.streaming_prefill(
            session_id=session_id,
            msgs=[user_msg],
            omni_mode=False,
            use_tts_template=True,
            is_last_chunk=True,
        )
        t_prefill = time.perf_counter() - t0

        iter_gen = ov_model.streaming_generate(
            session_id=session_id,
            generate_audio=True,
            use_tts_template=True,
            enable_thinking=False,
            do_sample=True,
            max_new_tokens=max_new_tokens,
        )

        SAMPLE_RATE = 24000
        audios = []
        full_text = ""

        # 流式播放：使用 sounddevice.OutputStream 边生成边播放
        stream = None
        if play_audio:
            try:
                import sounddevice as sd
                stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                stream.start()
                print("\n  [流式播放] 开始...", flush=True)
            except Exception as e:
                print(f"\n  [流式播放] 初始化失败，跳过播放：{e}")
                stream = None

        for wav_chunk, text_chunk in iter_gen:
            audios.append(wav_chunk)
            full_text += text_chunk
            if stream is not None:
                # wav_chunk shape: [1, samples]，转为 float32 mono
                chunk_np = wav_chunk[0].cpu().numpy().astype("float32")
                stream.write(chunk_np)

        if stream is not None:
            stream.stop()
            stream.close()
            print("  [流式播放] 完成", flush=True)

        total_s = time.perf_counter() - t0

        # 计算生成音频的总时长（秒）
        audio_duration_s = 0.0
        if audios:
            waveform = torch.cat(audios, dim=-1)[0]
            audio_duration_s = waveform.shape[-1] / SAMPLE_RATE
            if save_path:
                sf.write(save_path, waveform.cpu().numpy(), samplerate=SAMPLE_RATE)

        rtf = total_s / audio_duration_s if audio_duration_s > 0 else float("inf")

        num_tokens = count_tokens(ov_model, full_text)
        tps = num_tokens / total_s if total_s > 0 else 0.0

        return {
            "ttft_s": t_prefill,   # prefill 时间近似为 TTFT
            "total_s": total_s,
            "answer": full_text,
            "num_tokens": num_tokens,
            "tps": tps,
            "audio_duration_s": audio_duration_s,
            "rtf": rtf,
        }

    def warmup():
        _run_tts("你好", save_path=None, max_new_tokens=64)

    def bench():
        return _run_tts(question, save_path=output_audio_path, max_new_tokens=256)

    results = run_benchmark("文本输入 + 音频输出（TTS）", ov_model, warmup, bench, num_runs)

    # ---------- RTF 统计 ----------
    if results:
        rtf_list = [r["rtf"] for r in results if r.get("rtf", float("inf")) != float("inf")]
        dur_list = [r["audio_duration_s"] for r in results if r.get("audio_duration_s", 0) > 0]

        def avg(lst): return sum(lst) / len(lst) if lst else 0.0
        def mn(lst): return min(lst) if lst else 0.0
        def mx(lst): return max(lst) if lst else 0.0

        if rtf_list:
            print(f"\n  ── RTF（实时率）统计 ──")
            print(f"  音频时长   avg={avg(dur_list):.2f}s  min={mn(dur_list):.2f}s  max={mx(dur_list):.2f}s")
            print(f"  RTF        avg={avg(rtf_list):.3f}  min={mn(rtf_list):.3f}  max={mx(rtf_list):.3f}")
            rtf_avg = avg(rtf_list)
            if rtf_avg < 1.0:
                print(f"  ✅ 快于实时（RTF={rtf_avg:.3f} < 1.0，实时倍速={1/rtf_avg:.2f}x）")
            else:
                print(f"  ⚠️  慢于实时（RTF={rtf_avg:.3f} > 1.0，需优化 {(rtf_avg-1)*100:.1f}%）")

    if Path(output_audio_path).exists():
        print(f"\n  最后一次 TTS 音频已保存至：{output_audio_path}")
    else:
        print(f"  ⚠️ TTS 音频未生成（可能模型无音频输出）")


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 性能基准测试")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="主推理设备（默认 GPU）")
    parser.add_argument("--tts-device", default=DEFAULT_TTS_DEVICE, dest="tts_device",
                        help="TTS 推理设备（默认 GPU）")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"每个场景的测试次数（默认 {DEFAULT_RUNS}）")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["text", "image", "audio", "tts", "all"],
        default=["all"],
        help="要测试的场景（默认 all）",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="TTS 场景推理时实时流式播放音频（需要音频设备）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_all = "all" in args.scenarios
    scenarios = set(args.scenarios)

    print("=" * 65)
    print("  MiniCPM-o 4.5  OpenVINO  性能基准测试")
    print("=" * 65)
    print(f"  设备       : {args.device}")
    print(f"  TTS 设备   : {args.tts_device}")
    print(f"  每场景次数 : {args.runs}")
    print(f"  测试场景   : {args.scenarios}")
    print(f"  模型路径   : {ov_model_path}")
    print("=" * 65)

    # ---------- 加载模型 ----------
    from minicpm_o_4_5_helper import OVMiniCPMO

    print("\n加载模型...")
    t_load_start = time.perf_counter()
    ov_model = OVMiniCPMO(
        model_path=str(ov_model_path),
        device=args.device,
        tts_device=args.tts_device,
    )
    t_load_end = time.perf_counter()
    print(f"✅ 模型加载完成（耗时 {t_load_end - t_load_start:.1f}s）")

    # 场景 4 需要 TTS，提前初始化
    need_tts = run_all or "tts" in scenarios
    if need_tts:
        print("\n初始化 TTS 模块...")
        t_tts_start = time.perf_counter()
        ov_model.init_tts()
        t_tts_end = time.perf_counter()
        print(f"✅ TTS 初始化完成（耗时 {t_tts_end - t_tts_start:.1f}s）")

    # ---------- 运行各场景 ----------
    if run_all or "text" in scenarios:
        benchmark_text_only(ov_model, args.runs)

    if run_all or "image" in scenarios:
        benchmark_text_image(ov_model, args.runs)

    if run_all or "audio" in scenarios:
        benchmark_text_audio(ov_model, args.runs)

    if run_all or "tts" in scenarios:
        benchmark_text_audio_output(ov_model, args.runs, play_audio=args.play)

    print("\n✅ 全部基准测试完成。")


if __name__ == "__main__":
    main()
