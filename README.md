# MyBot

运行在 RTX 4060 Laptop 8 GB 上的中文流式语音助手。本机完成 VAD、ASR 和 TTS，远端 LLM 负责回复，Windows 浏览器负责麦克风与播放。

```text
浏览器麦克风 -> VAD -> Qwen3-ASR -> DeepSeek token 流
                   -> 短句队列 -> qwentts.cpp -> 浏览器播放 / 渲染确认
```

## 快速运行

当前机器已配置好 `mybot` 环境：

```bash
./run.sh start
```

在 **Windows Chrome/Edge** 打开 **http://127.0.0.1:8765**，点击“开启麦克风”。冷启动通常需要 2-4 分钟；终端关闭不影响后台运行，重复启动不会重复加载模型。

```bash
./run.sh status
./run.sh logs
./run.sh stop
```

默认只监听回环地址，同时允许一个浏览器会话。端口冲突使用 `--port 8766`，其他解释器通过 `MYBOT_PYTHON` 指定。默认轮流对话，按钮随时可打断；戴耳机时可设置 `BARGE_IN=true` 启用语音打断。

## 首次安装

依赖只有一个入口：[`pyproject.toml`](pyproject.toml)。目前支持 Python 3.12、Linux/WSL 和 NVIDIA CUDA，不要另装独立 `flash-attn` wheel。

```bash
conda create -n mybot python=3.12 -y
conda activate mybot
sudo apt-get install -y libportaudio2 portaudio19-dev
python -m pip install -e .
python -m qwentts_cpp doctor
python -m qwentts_cpp prefetch --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --quant Q8_0
cp .env.example .env
chmod 600 .env
```

已有 `.env` 时不要覆盖。填写 `DEEPSEEK_API_KEY`，并按 [assets/README.md](assets/README.md) 准备参考音频。启动器也支持隐藏输入缺失的 API Key。ASR 首次启动需要下载模型，TTS 使用上一步预取的模型缓存。

这是源码检出的部署项目，推荐 editable 安装并保留此目录。模型保留在 Hugging Face 缓存，密钥、录音、模型权重和运行产物不进入 Git。

## 代码地图

| 位置 | 唯一职责 |
| --- | --- |
| `core/app.py` | 装配、预热和释放模型及音频资源 |
| `core/orchestrator.py` | 一轮对话的并发、取消、背压和历史 |
| `core/config.py` / `types.py` | 环境配置、PCM 数据和阶段计时 |
| `core/asr/` | VAD、流式识别会话、音频分帧调度 |
| `core/llm/` | 可取消的云端 token 流、过滤与短句切分 |
| `core/tts/` | 原生 Qwen TTS 与有界 PCM 生成 |
| `core/audio/` | 浏览器/直接设备播放、麦克风、统一播放接口 |
| `core/web/` | HTTP/WebSocket 服务及 `static/` 网页 |
| `core/cli.py` / `main.py` | 前台文字/直接音频设备调试入口 |
| `scripts/` | 后台进程管理、音频设备检查 |
| `tests/unit/` | 不加载 GPU 模型、不访问云 API 的回归测试 |
| `tests/integration/` | 显式运行的真实模型与浏览器链路验收 |

建议阅读顺序：`config.py` → `types.py` → `app.py` → `orchestrator.py`，再进入具体模型或音频模块。模型层不负责打开设备，调度层通过接口使用合成器和播放器。

## 开发与验收

```bash
python -m pip install -e '.[dev,browser]'
python -m playwright install chromium
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

`dev` 提供 pytest/Ruff，`browser` 提供集成验收工具，不增加另一份依赖清单。普通开发只需 `.[dev]`。真实模型验收会占用 GPU 和调用付费 API，不随 pytest 自动运行。

- [架构与配置](docs/architecture.md)：模块边界、数据流、取消与背压。
- [开发与验收](docs/development.md)：集成测试、排障、贡献约定与 Git 清理边界。
- [模型选型与基准](docs/benchmarks.md)：保留实验结论，不把旧实验代码混入运行时。
