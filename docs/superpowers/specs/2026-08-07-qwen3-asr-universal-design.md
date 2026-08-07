# Qwen3-ASR-Universal 设计文档

日期: 2026-08-07
状态: 已批准

## 1. 背景与目标

当前字幕翻译 skill 依赖 `transcribe.exe`(HaujetZhao/Qwen3-ASR-GGUF v0.1 tag 的 PyInstaller 产物), 仅支持 Windows, 且 GPU 加速只接了 DirectML(Windows 独占)和 Vulkan(README 只写了 Windows 接入)。

目标: 写一个**跨平台 Qwen3-ASR 命令行工具**, 在 Windows / macOS / Linux 上运行, 支持 CPU / CUDA / Vulkan / Metal 硬件加速, 输出与 `transcribe.exe` 一致的词级时间戳 JSON(`[{text, start, end}]`, 秒), 作为 skill 的通用替代运行时。

非目标(YAGNI):
- 不照抄上游代码(上游仓库无 LICENSE), 只复用 Apache-2.0 的模型权重。
- 不做流式转录、不做 vllm/transformers 后端、不做 ITN 之外的额外后处理。
- 首发不做 MLX(Mac 用 Metal)、不做 macos-x64 构建。

## 2. 关键决策(已确认)

| 决策点 | 结论 |
|---|---|
| 项目形态 | 独立新仓库 `qwen3-asr-universal`, 与 skill 仓库分离 |
| 技术栈 | Python; onnxruntime(encoder) + llama-cpp-python(decoder/aligner) |
| 路线 | 路线 A: 全 Python 移植 v0.1 管线, 转录引擎抽象成接口(v2 可换 llama.cpp 原生) |
| 发布形态 | GitHub Actions 矩阵构建, 按 `平台-架构-后端` 出 zip, 带 manifest.json |
| Mac 加速 | Metal(llama.cpp 默认); onnxruntime-silicon(MPS) 跑 encoder |
| 默认模型 | Qwen3-ASR-1.7B(int4 encoder + q4_k decoder), 首次运行自动下载 |
| 时间戳 | 依赖 Qwen3-ForcedAligner-0.6B(词级), 对齐器是必需组件 |
| 模型来源 | 复用 HaujetZhao 已发布的转换产物(权重 Apache-2.0), 不自行导出 |
| 输出 | 词级 JSON(必需) + .txt/.srt(兼容 skill 现有接口) |
| 输出管道 | 全程 UTF-8, 状态只打 ASCII, 无 emoji → 杜绝 Windows GBK 崩溃, 不需要 CREATE_NEW_CONSOLE |

## 3. 系统架构

```
qwen3-asr-universal/
├── pyproject.toml              # 包定义 + CLI 入口 (q3asr)
├── q3asr/
│   ├── __main__.py             # CLI 入口
│   ├── cli.py                  # 参数解析
│   ├── audio.py                # 任意音频 → 16k 单声道 (imageio-ffmpeg 静态 ffmpeg)
│   ├── features.py             # 对数 Mel 特征 (numpy/scipy)
│   ├── encoder.py              # ONNX encoder 封装 (onnxruntime)
│   ├── decoder.py              # GGUF decoder 封装 (llama-cpp-python)
│   ├── aligner.py              # 强制对齐 → 词级时间戳
│   ├── engine.py               # 编排: 分块 → 编码 → 解码 → 对齐 (子进程并行)
│   ├── transcription.py        # 转录引擎抽象接口 (v2 换 llama.cpp 原生引擎的位置)
│   ├── backend.py              # 设备探测: cuda→vulkan→metal→cpu
│   ├── models.py               # 模型清单 + 自动下载/缓存/校验
│   └── output.py               # 写 .json/.txt/.srt
├── resources/
│   └── models.yaml             # 模型清单 (URL/大小/sha256/镜像)
├── ci/
│   └── build.py                # PyInstaller 打包脚本 (BUILD_TARGET 环境变量选后端)
├── .github/workflows/release.yml
└── tests/
```

### 3.1 CLI 接口(与 skill 现有调用完全兼容)

```
q3asr <audio> [-y] [--seek-start X] [--duration Y] [-l language]
             [--prec fp32|fp16|int8|int4] [--device auto|cuda|vulkan|metal|cpu]
             [--model 1.7b|0.6b] [--model-dir PATH] [--no-dml] [--no-vulkan] [--json-only]
```

- `--no-dml` / `--no-vulkan` 为兼容旧参数接受并忽略(映射为设备回退)。
- 输出写到输入音频同目录: `<basename>.json`(词级时间戳, 必须)、`<basename>.txt`、`<basename>.srt`(兼容)。`--json-only` 可只写 JSON。
- 时间戳为**秒**浮点; `--seek-start/--duration` 时, 输出时间戳**相对切片起点**(与现 exe 一致, 由 skill 用 `timestamp_to_yaml.py --offset` 恢复绝对时间)。
- 退出码: 0 成功; 1 通用错误; 2 参数错误; 3 模型下载/加载失败。

### 3.2 推理管线(子进程并行)

参考 v0.1 的主进程 + 辅助进程架构, 规避 Python GIL:

```
主进程: 读音频 → 分块(默认 30s) → 调度
  ├── 编码子进程: features(fbank) → ONNX encoder → embedding
  └── 对齐子进程: 对齐器 ONNX encoder + GGUF decoder → 词级时间戳
主进程: decoder(GGUF) 逐块解码 → 文本
```

- 长音频分块处理, 块间保留上下文避免断句。
- `--seek-start/--duration` 在解码阶段实现(先切音频再走管线)。

### 3.3 设备选择

- 优先级: `--device` > env `QASR_DEVICE` > 自动探测(cuda → vulkan → metal → cpu)。
- 自动探测: 尝试 import 对应后端, 失败即降级, 打印 `[INFO] using backend: <name>`。
- 每个发布 zip 内置其对应后端的 onnxruntime 与 llama-cpp-python wheel; zip 内程序检测到 GPU 后端不可用则告警并回退 CPU(不报错)。

## 4. 发布矩阵与 CI

### 4.1 发布产物(首发)

| zip | onnxruntime (encoder) | llama-cpp-python (decoder) |
|---|---|---|
| `qwen3-asr-windows-x64-cpu.zip` | onnxruntime | cpu |
| `qwen3-asr-windows-x64-cuda.zip` | onnxruntime-gpu | cuda |
| `qwen3-asr-windows-x64-vulkan.zip` | onnxruntime-directml | vulkan |
| `qwen3-asr-linux-x64-cpu.zip` | onnxruntime | cpu |
| `qwen3-asr-linux-x64-cuda.zip` | onnxruntime-gpu | cuda |
| `qwen3-asr-linux-x64-vulkan.zip` | onnxruntime | vulkan |
| `qwen3-asr-macos-arm64-metal.zip` | onnxruntime-silicon | metal |
| `qwen3-asr-macos-arm64-cpu.zip` | onnxruntime | cpu |

- zip 内容: PyInstaller 打包的可执行 `q3asr` + `manifest.json` + `imageio-ffmpeg` 静态 ffmpeg(自动打入)。
- `manifest.json`: `{os, arch, backend, version, files:[{name, sha256, size}], cli:"q3asr"}`。
- GitHub Actions 矩阵构建, 每个 zip 为独立 job; 打 tag 触发 `release.yml` 自动发布。

### 4.2 模型来源与下载

模型文件(首次运行自动下载到 `~/.cache/q3asr/models/`, 断点续传 + sha256 校验):

| 组件 | 文件(zip 内含) | 大小 | 来源 |
|---|---|---|---|
| ASR 1.7B(默认) | encoder frontend/backend `.int4.onnx`, decoder `.q4_k.gguf`, mel filters | ~1.4 GB | HaujetZhao models release |
| Aligner 0.6B | aligner encoder `.int4.onnx` + aligner decoder `.gguf` | ~505 MB | 同上 |
| ASR 0.6B(可选) | 同上(0.6B) | ~564 MB | 同上 |

- `resources/models.yaml`: 每文件的 URL、sha256、大小、解压目标。
- 镜像: `QASR_MODEL_MIRROR=gh|ms`。默认 `gh`(GitHub release); `ms` 指向 Modelscope(需先上传镜像, 首发若未上传则仅 `gh`)。
- 首次运行输出下载进度与总大小提示, 便于用户决定。

## 5. skill 集成

修改 `subtitle-translate-skill`(skill 仓库):

- `run_transcribe.py` 增加自动安装逻辑(现有 `TRANSCRIBE_EXE` 手动指定优先级最高):
  1. 首次需要转录时, 请求 ASR 仓库最新 release 的 `manifest.json`, 按 os/arch + `TRANSCRIBE_BACKEND`(默认 auto)选 zip。
  2. 下载 → sha256 校验 → 解压到 `~/.cache/opencode-translate/asr/<version>/`。
  3. 之后直接复用缓存; 支持 `TRANSCRIBE_ASR_VER` 锁定版本。
- 新环境变量: `TRANSCRIBE_BACKEND=cuda|vulkan|metal|cpu`(默认 auto); `TRANSCRIBE_MODEL_MIRROR`。
- `SKILL.md` 步骤 1 更新: 首次从视频/音频开始时自动获取 ASR 运行时与模型。
- 与 `timestamp_to_yaml.py` 的 JSON 契约不变。

## 6. 测试策略

- 单测(unittest, 无模型依赖):
  - `features`: mel/fbank 数值正确性、采样率/单声道归一化。
  - `cli`: 参数解析与 `build_cmd` 兼容(与 skill `run_transcribe.py` 构造的命令逐项对照)。
  - `backend`: 探测顺序、`--device` 覆盖、env 覆盖。
  - `models`: models.yaml 解析、sha256 校验、镜像 URL 切换。
  - `output`: JSON schema(`[{text,start,end}]`, start<end, 递增, 秒)。
- 集成测试(标记者, 需下载模型, CI 只跑 CPU + 0.6B):
  - 30s 音频 → 词级 JSON, 断言 schema、文本非空、时间落在音频时长内。
- 回归: skill 现有 24 个测试在改动后保持通过。

## 7. 兼容性与已知风险

- 上游转换产物的 ONNX/GGUF 文件为平台无关格式, 可在三平台直接复用。
- `onnxruntime-silicon`(MPS)对全部 encoder 算子支持需实测, 若不支持则 mac metal zip 的 encoder 回退 CPU(decoder 仍 Metal)。
- `llama-cpp-python` 后端 wheel 需与 Python/onnxruntime 版本对齐, CI 锁定依赖版本。
- GitHub release 下载在国内可能慢/失败, 是 v1 主要痛点; Modelscope 镜像为 v1.1 增强项。
- 上游仓库无 LICENSE, 本项目**自行编写全部代码**, 仅引用模型权重(Apache-2.0); 项目采用 Apache-2.0 许可。
