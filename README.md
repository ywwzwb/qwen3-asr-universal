# Qwen3-ASR-Universal

跨平台的 [Qwen3-ASR](https://github.com/HaujetZhao/Qwen3-ASR-GGUF) 命令行工具,输出**词级时间戳 JSON**,同时生成 `.txt` / `.srt` 字幕。Windows / macOS / Linux 三平台,支持 CPU / CUDA / Vulkan / Metal 硬件加速,开箱即用(无需安装 Python 环境)。

## 特性

- **跨平台**: Windows x64、Linux x64、macOS (arm64)
- **硬件加速**: CPU / CUDA / Vulkan / Metal,自动探测、失败自动回退 CPU
- **模型自动下载**: 首次运行自动下载 Qwen3-ASR 模型与强制对齐器(断点续传 + sha256 校验)
- **词级时间戳**: 输出 `[{text, start, end}]`(秒),可直接用于字幕生成/对齐
- **轻量发布**: PyInstaller 单文件打包,下载解压即用

## 安装

从 [GitHub Releases](https://github.com/ywwzwb/qwen3-asr-universal/releases) 下载对应平台的 zip,解压后使用 `q3asr`(Windows 为 `q3asr.exe`):

| 平台 | 文件 | 说明 |
|---|---|---|
| Windows x64 | `qwen3-asr-windows-x64-cpu.zip` / `-cuda` / `-vulkan` | CPU / NVIDIA CUDA / DirectML(Vulkan) |
| Linux x64 | `qwen3-asr-linux-x64-cpu.zip` / `-cuda` / `-vulkan` | CPU / NVIDIA CUDA / Vulkan |
| macOS arm64 | `qwen3-asr-macos-arm64-metal.zip` / `-cpu` | Metal(Apple Silicon)/ CPU |

> Linux 版本在 Ubuntu 20.04(glibc ≥ 2.31)上构建,兼容所有更新的发行版。

## 快速开始

```bash
# 转录 audio.mp3,在相同目录生成 audio.json / audio.txt / audio.srt
./q3asr audio.mp3

# 只输出词级时间戳 JSON
./q3asr audio.mp3 --json-only

# 只下载模型(不转录),便于提前准备
./q3asr --download-models-only

# 指定模型与设备
./q3asr audio.mp3 --model 0.6b --device cuda
```

首次运行会下载模型到 `~/.cache/q3asr/models/`(可用 `Q3ASR_CACHE_DIR` 覆盖),约 1.4 GB(1.7B)。

### 输出

- `<basename>.json` — 词级时间戳 `[{text, start, end}]`,秒
- `<basename>.txt` — 纯文本
- `<basename>.srt` — SRT 字幕

## CLI 参数

```
q3asr <audio> [-y] [--seek-start X] [--duration Y] [-l language]
             [--prec int4] [--device auto|cuda|vulkan|metal|cpu]
             [--model 1.7b|0.6b] [--model-dir PATH] [--no-dml] [--no-vulkan]
             [--json-only] [--download-models-only]
```

| 参数 | 说明 |
|---|---|
| `input` | 音频文件(mp3 / wav / mkv / 任意 ffmpeg 可解码格式) |
| `--seek-start` / `--duration` | 只转录切片,时间戳相对切片起点(秒) |
| `-l, --language` | 语言提示(如 `zh`、`en`) |
| `--prec` | 精度,默认 `int4` |
| `--device` | `auto`(默认)/ `cuda` / `vulkan` / `metal` / `cpu`;可用环境变量 `QASR_DEVICE` 覆盖 |
| `--model` | `1.7b`(默认)/ `0.6b` |
| `--model-dir` | 使用本地平铺模型目录(不自动下载) |
| `--n-ctx` | LLM 上下文窗口(KV cache 大小),默认 2048 |
| `--no-dml` / `--no-vulkan` | 兼容旧参数,忽略 |
| `--json-only` | 只写 `.json` |
| `--download-models-only` | 只下载模型后退出 |
| `-y, --yes` | 跳过交互确认 |

退出码: `0` 成功,`1` 通用错误,`2` 参数错误,`3` 模型下载/加载失败。

## 模型

模型来自 [HaujetZhao/Qwen3-ASR-GGUF](https://github.com/HaujetZhao/Qwen3-ASR-GGUF)(权重 Apache-2.0),由 `resources/models.yaml` 描述 URL、sha256 与大小:

| 组件 | 大小 | 说明 |
|---|---|---|
| ASR 1.7B(默认) | ~1.4 GB | encoder int4 ONNX + decoder q4_k GGUF |
| ASR 0.6B | ~564 MB | 更轻量,精度略低 |
| Aligner 0.6B | ~505 MB | 强制对齐器,词级时间戳必需,自动下载 |

- 缓存目录: `~/.cache/q3asr/models/`(可用 `Q3ASR_CACHE_DIR` 修改)
- 断点续传下载 + sha256 校验,损坏自动重下
- 镜像切换(开发者): `resources/models.yaml` 中 `base_urls` 的 `gh`(GitHub)/ `ms`(Modelscope)

## 设备选择

优先级: `--device` > 环境变量 `QASR_DEVICE` > 自动探测(`cuda` → `vulkan` → `metal` → `cpu`)。

每个发布 zip 内置对应后端的推理库;检测到 GPU 不可用时会告警并回退 CPU(不报错)。

## 从源码运行

```bash
pip install -e .
q3asr audio.mp3
```

Python ≥ 3.10。运行测试:

```bash
pip install pytest
pytest
```

## 构建发布包

```bash
python ci/build.py linux-x64-cpu      # 可选: windows-x64-cpu|cuda|vulkan
                                     #        linux-x64-cpu|cuda|vulkan
                                     #        macos-arm64-metal|cpu
```

打 tag(`v*`)自动触发 GitHub Actions 矩阵构建并发布到 Releases:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

## 许可

[MIT](./LICENSE)
