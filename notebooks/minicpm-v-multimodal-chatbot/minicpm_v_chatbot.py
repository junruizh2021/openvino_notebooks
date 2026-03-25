"""
MiniCPM-V Multimodal Chatbot with OpenVINO
使用 OpenVINO 运行 MiniCPM-V 多模态聊天机器人

## 依赖安装

# 1. PyTorch（CPU 版，转换用）
pip install "torch==2.8" "torchvision==0.23.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 2. 基础依赖
pip install \
    "transformers==4.53.3" \
    "timm>=0.9.2" \
    "sentencepiece" \
    "peft" \
    "huggingface-hub>=0.24.0" \
    "nncf>=2.14.0" \
    "Pillow" \
    "numpy<2.0" \
    "requests" \
    "gradio>=4.40,<6"

# 3. 先安装 optimum（minicpm4v 定制分支）
pip install \
    "git+https://github.com/openvino-dev-samples/optimum.git@minicpm4v" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 4. 再安装 optimum-intel（--no-deps 跳过版本冲突）
pip install \
    "git+https://github.com/openvino-dev-samples/optimum-intel.git@81f69be5e0f7fd80aba81b5e5d4c213dfe3e9344" \
    --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 5. OpenVINO 推理三件套
pip install -U --pre \
    "openvino>=2025.0" \
    "openvino-tokenizers>=2025.0" \
    "openvino-genai>=2025.0"

用法:
    # 使用默认本地模型转换并启动 Gradio Demo
    python minicpm_v_chatbot.py

    # 指定其他本地模型路径
    python minicpm_v_chatbot.py --model /path/to/your/model

    # 指定 HuggingFace 在线模型
    python minicpm_v_chatbot.py --model openbmb/MiniCPM-V-4_5

    # 指定推理设备
    python minicpm_v_chatbot.py --device CPU

    # 仅做单次推理（不启动 Gradio）
    python minicpm_v_chatbot.py --image cat.png --question "What is unusual on this image?"

    # 跳过模型转换（OpenVINO 模型目录已存在时使用）
    python minicpm_v_chatbot.py --skip-convert
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image


# ── 模型配置 ────────────────────────────────────────────────────────────────────
# 客户提供的本地模型路径（优先使用）
LOCAL_MODEL_PATH = "/home/nvme-data/AI-models/MiniCPM-V-4"

# HuggingFace 在线模型备选列表
HF_MODEL_IDS = [
    "openbmb/MiniCPM-V-4_5",
    "openbmb/MiniCPM-V-4",
    "openbmb/MiniCPM-V-2_6",
]

DEFAULT_MODEL = LOCAL_MODEL_PATH
DEFAULT_DEVICE = "AUTO"
DEFAULT_MAX_NEW_TOKENS = 128


# ── 工具函数 ────────────────────────────────────────────────────────────────────

def download_example_images():
    """示例图片下载（已禁用，网络环境不支持访问 GitHub）。"""
    pass


def load_image(image_file: str):
    """加载图片并返回 PIL Image 和 OpenVINO Tensor。"""
    import openvino as ov

    if isinstance(image_file, str) and (
        image_file.startswith("http") or image_file.startswith("https")
    ):
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")

    image_data = (
        np.array(image.getdata())
        .reshape(1, image.size[1], image.size[0], 3)
        .astype(np.uint8)
    )
    return image, ov.Tensor(image_data)


def get_model_name(model_id: str) -> str:
    """从模型 ID 或本地路径中提取简短名称。"""
    return Path(model_id).name


def get_ov_model_dir(model_id: str) -> Path:
    """根据模型 ID 或本地路径推导 OpenVINO 输出目录。"""
    candidate = Path(model_id)
    # 本地模型：保存到“原始路径的父目录/xxx-ov-int4”
    if candidate.is_absolute() or candidate.exists():
        resolved = candidate.expanduser().resolve()
        return resolved.parent / f"{resolved.name}-ov-int4"
    # 在线模型：在当前目录下创建 xxx-ov-int4
    return Path(f"{get_model_name(model_id)}-ov-int4")


def is_conversion_complete(model_dir: Path) -> bool:
    """检查 OpenVINO 转换输出是否完整（模型文件和 tokenizer 文件都存在）。"""
    required_files = [
        "openvino_language_model.xml",
        "openvino_language_model.bin",
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
    ]
    return all((model_dir / f).exists() for f in required_files)


def convert_model(model_id: str, model_dir: Path):
    """使用 optimum-cli 将模型转换为 OpenVINO IR（INT4 量化）。

    model_id 可以是 HuggingFace ID（如 openbmb/MiniCPM-V-4_5）
    或本地模型路径（如 /home/nvme-data/AI-models/minicpm-v/checkpoint-2400）。
    """
    if model_dir.exists() and is_conversion_complete(model_dir):
        print(f"OpenVINO 模型已存在且完整，跳过转换: {model_dir}")
        return

    if model_dir.exists() and not is_conversion_complete(model_dir):
        print(f"警告：检测到残缺的转换目录 {model_dir}，将重新转换...")
        import shutil
        shutil.rmtree(model_dir)
    else:
        model_dir.mkdir(parents=True, exist_ok=True)

    # 本地路径校验
    model_path = Path(model_id)
    if model_path.is_absolute() or model_path.exists():
        if not model_path.exists():
            print(f"错误：本地模型路径不存在: {model_id}")
            sys.exit(1)
        print(f"正在转换本地模型: {model_id} -> {model_dir}")
    else:
        print(f"正在从 HuggingFace 下载并转换模型: {model_id} -> {model_dir}")

    print("调用 optimum-intel Python API 进行转换（INT4 量化）...")
    from optimum.exporters.openvino.__main__ import main_export
    from optimum.intel.openvino.configuration import OVConfig, OVWeightQuantizationConfig
    import nncf

    # 本模型视觉编码器 fc2 层通道数为 4304（不能被 128 整除），
    # NNCF 3.1.0 不支持 group_size_fallback_mode，用 ignored_scope 跳过这些层，
    # 使其保持 int8（backup_mode 默认）而非 int4 分组量化
    # validate=False：各子模型单独量化时，pattern 在 LLM 子图中不存在不报错，
    # 仅在 vision encoder 子图中匹配生效
    ignored_scope = nncf.IgnoredScope(
        patterns=[".*encoder\\.layers\\..*\\.mlp\\.fc2.*"],
        validate=False,
    )
    quantization_config = OVWeightQuantizationConfig(
        bits=4,
        group_size=128,
        ratio=0.8,
        ignored_scope=ignored_scope,
    )
    ov_config = OVConfig(quantization_config=quantization_config)

    main_export(
        model_name_or_path=str(model_id),
        output=model_dir,
        task="image-text-to-text",
        trust_remote_code=True,
        ov_config=ov_config,
        convert_tokenizer=True,
    )


# ── Gradio Demo ─────────────────────────────────────────────────────────────────

def make_demo(model, model_name: str):
    """构建 Gradio 多模态聊天 Demo。"""
    import inspect

    import gradio as gr
    import openvino as ov
    import openvino_genai as ov_genai

    has_additional_buttons = (
        "undo_button" in inspect.signature(gr.ChatInterface.__init__).parameters
    )

    def read_image(path: str) -> ov.Tensor:
        pic = Image.open(path).convert("RGB")
        image_data = (
            np.array(pic.getdata())
            .reshape(1, pic.size[1], pic.size[0], 3)
            .astype(np.uint8)
        )
        return ov.Tensor(image_data)

    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")

        generation_config = ov_genai.GenerationConfig()
        generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS

        files = message["files"] if isinstance(message, dict) else message.files
        message_text = message["text"] if isinstance(message, dict) else message.text

        if not history:
            model.start_chat()

        image = None
        if files:
            if isinstance(files[-1], dict):
                image = files[-1]["path"]
            else:
                if isinstance(files[-1], (str, Path)):
                    image = files[-1]
                else:
                    image = (
                        files[-1]
                        if isinstance(files[-1], (list, tuple))
                        else files[-1].path
                    )
        if image is not None:
            image = read_image(image)

        # VLMPipeline 非线程安全，在当前线程内同步调用 generate()
        # streamer_cb 每收到一个 token 追加到 buffer
        buffer = []

        def streamer_cb(token: str) -> bool:
            buffer.append(token)
            return False  # False=继续生成，True=停止

        generation_kwargs = {
            "prompt": message_text,
            "generation_config": generation_config,
            "streamer": streamer_cb,
        }
        if image is not None:
            generation_kwargs["image"] = image

        model.generate(**generation_kwargs)
        yield "".join(buffer)

    additional_buttons = {}
    if has_additional_buttons:
        additional_buttons = {"undo_button": None, "retry_button": None}

    demo = gr.ChatInterface(
        fn=bot_streaming,
        title=f"{model_name} OpenVINO Chatbot",
        examples=[
            {"text": "What is on the flower?", "files": ["./bee.jpg"]},
            {"text": "How to make this pastry?", "files": ["./baklava.png"]},
        ],
        stop_btn=None,
        multimodal=True,
        **additional_buttons,
    )
    return demo


# ── 单次推理 ─────────────────────────────────────────────────────────────────────

def run_single_inference(ov_model, image_path: str | None, question: str):
    """执行一次推理并打印结果。image_path 为 None 时做纯文本推理。"""
    import openvino_genai as ov_genai

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    image_tensor = None
    if image_path:
        _, image_tensor = load_image(image_path)

    print(f"\n问题: {question}\n回答: ", end="", flush=True)

    def streamer(subword: str) -> bool:
        print(subword, end="", flush=True)

    ov_model.start_chat()
    generate_kwargs = {"generation_config": config, "streamer": streamer}
    if image_tensor is not None:
        generate_kwargs["image"] = image_tensor
    ov_model.generate(question, **generate_kwargs)
    print()


# ── 入口 ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="MiniCPM-V Multimodal Chatbot with OpenVINO"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"本地模型路径或 HuggingFace 模型 ID（默认: {DEFAULT_MODEL}）\n"
            f"HF 可选: {HF_MODEL_IDS}"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"推理设备，例如 CPU / GPU / AUTO（默认: {DEFAULT_DEVICE}）",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="跳过模型转换（模型目录已存在时使用）",
    )
    parser.add_argument(
        "--image",
        default=None,
        nargs="?",
        const="",
        help="单次推理时使用的图片路径（可不填，纯文本推理时省略路径即可）",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="单次推理时的问题；传入此参数即进入单次推理模式（不启动 Gradio）",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Gradio 服务器地址（远程部署时使用）",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Gradio 服务器端口（远程部署时使用）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="通过 Gradio 公共链接分享 Demo",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir  = get_ov_model_dir(args.model)
    model_name = get_model_name(args.model)

    # 1. 下载示例图片
    download_example_images()

    # 2. 转换模型（如未跳过）
    if not args.skip_convert:
        convert_model(args.model, model_dir)
    else:
        if not model_dir.exists() or not is_conversion_complete(model_dir):
            print(f"错误：OpenVINO 模型目录不存在或不完整 ({model_dir})，请先移除 --skip-convert 重新转换。")
            sys.exit(1)

    # 3. 加载 OpenVINO 模型
    import openvino_genai as ov_genai

    print(f"\n加载模型: {model_dir}  设备: {args.device}")
    ov_model = ov_genai.VLMPipeline(str(model_dir), device=args.device)
    print("模型加载完成。\n")

    # 4. 单次推理 或 启动 Gradio Demo
    # 传入 --question 即进入单次推理模式（--image 可选）
    if args.question is not None:
        question = args.question
        image_path = args.image if args.image else None
        run_single_inference(ov_model, image_path, question)
    else:
        demo = make_demo(ov_model, model_name)

        launch_kwargs = {"debug": True, "height": 600, "server_name": "0.0.0.0"}
        if args.server_name:
            launch_kwargs["server_name"] = args.server_name
        if args.server_port:
            launch_kwargs["server_port"] = args.server_port
        if args.share:
            launch_kwargs["share"] = True

        try:
            demo.launch(**launch_kwargs)
        except Exception:
            launch_kwargs["share"] = True
            demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
