# RTX 4060 语音模型选型与基准

测试日期：2026-09-18。硬件为 GeForce RTX 4060 Laptop 8 GB，Python 3.12。数字用于本项目单并发交互选型，不应外推为跨硬件排行榜。

## VAD 选型

[Silero VAD](https://github.com/snakers4/silero-vad) 的神经网络模型约 2 MB，官方报告单个 30 ms chunk 在单 CPU 线程低于 1 ms，噪声场景下值得作为后续质量升级。但当前瓶颈是端点等待而非 VAD 计算，本项目优先使用无额外模型状态和 ONNX 调度的 WebRTC VAD，并把默认端点从常见的 500–800 ms 压到 280 ms；160 ms pre-roll 用于补回触发窗口前的音素。

## TTS 调研结论

近期候选包括 Qwen3-TTS、Fun-CosyVoice3、MOSS-TTS-Realtime、VibeVoice-Realtime、Breeze TTS 2 和 Luna-TTS Realtime。

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 官方提供 0.6B/1.7B、中文和 3 秒音色克隆，并报告最低 97 ms 流式延迟。但官方 Python `generate_*` 路径在本机不能及时返回 PCM，性能远低于实时。
- [Fun-CosyVoice3](https://github.com/QwenAudio/CosyVoice) 支持文本/音频双流和中文音色克隆，官方报告最低 150 ms。其完整 PyTorch/vLLM 组合在本机首包仍明显偏慢。
- [MOSS-TTS-Realtime](https://github.com/OpenMOSS/MOSS-TTS) 是 1.7B + 200M local transformer，官方报告 180 ms TTFB；质量和上下文设计有吸引力，但对 ASR/TTS 共存的 8 GB 预算更激进。
- [VibeVoice](https://github.com/microsoft/VibeVoice) Realtime 0.5B 偏英语实时会话，不作为中文音色克隆主路径。
- [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) 面向中英双语且官方 H100 数字很快，但公开部署建议和显存需求不适合 8 GB。
- [Luna-TTS](https://arxiv.org/abs/2608.11593) 报告 41.6 ms warm first block 和 0.024 RTF，但当前缺少可直接复现的官方本地权重/运行时，暂不纳入工程依赖。

最终采用 RealtimeTTS 的 [native QwenEngine](https://github.com/KoljaB/RealtimeTTS)：它通过 `qwentts.cpp` 直接运行 Q8 GGUF 并逐块暴露 PCM，保留 Qwen3-TTS 0.6B Base 的中文音色克隆能力。

## TTS 本机结果

| 路径 | 首音频/首包 | RTF 或整段耗时 | 结论 |
| --- | ---: | ---: | --- |
| Qwen3-TTS 0.6B Base，官方 Python BF16 | 短句整段 6.92 s | 中文句子 24.98 s | 淘汰 |
| CosyVoice3 0.5B，PyTorch | 4.7–16.9 s | RTF 2.1–3.2 | 淘汰 |
| CosyVoice3 0.5B，vLLM warm | 1.805 s | RTF 0.544 | 仍不足以做主路径 |
| Kokoro | 0.20–0.29 s 整段 | 小模型快速 | 仅保留比较结果，已移除实验后端 |
| Qwen3-TTS native Q8 | engine 25–30 ms；可播放 136–188 ms | RTF 0.239–0.272 | 选用 |

native Qwen 路径在 `nvidia-smi` 中总占用约 3.4 GiB。可播放数字包含 80 ms startup buffer；减小该值能进一步提前发声，但更容易出现 underrun。上述路径使用同一个 Qwen Base 模型时，Python 与 native 的差异主要来自运行时和真正的增量 PCM 输出，而非更换模型。

## ASR 选型

[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 0.6B 支持 52 种语言/方言，官方 Python SDK 的流式推理要求 vLLM。初期 Torch 2.8 环境中，transformers + FlashAttention 2 单模型离线结果为：

| 音频长度 | warm 转写耗时 |
| ---: | ---: |
| 3 s | 1.449 s |
| 5 s | 1.785 s |
| 15 s | 3.45 s |

`torch.compile(dynamic=True)` 将 3 秒样本降到约 1.258 秒，但首次编译约 6.2 秒，收益不足以默认开启。transformers 峰值约 1.59 GiB torch allocation，进程级 GPU 占用约 2.93 GiB。

vLLM 经 8 GB 定向收缩（`gpu_memory_utilization=0.46`、`max_model_len=2048`、单序列、eager）后，单模型 warm 的 3/5/15 秒音频分别为 0.855/1.103/2.285 秒。早期只用静音预热时，共驻首轮出现过 3.251 秒 ASR 和 547 ms TTS 首 PCM；分析后确认静音没有覆盖真实语音的完整 kernel 路径。改用 3 秒参考语音预热后的公平结果为：

| ASR 后端 | 3 秒转写 | TTS 首 PCM | TTS RTF | 冷启动 |
| --- | ---: | ---: | ---: | ---: |
| transformers SDPA | 1.361 s | 185.1 ms | 0.273 | 较短 |
| vLLM eager | 0.823 s | 106.9 ms | 0.263 | ASR 约 74 s |

vLLM 还可在 VAD 说话阶段维护 streaming state。按真实 20 ms 帧速率喂入同一段 3 秒音频，端点后的 finalize 为 578.6 ms，转写与离线结果一致。因此它被选为 RTX 4060 默认；transformers 保留为启动更快、依赖更简单的回退。

在当前 Torch 2.9.1 环境中，用同一段 3 秒音频重复 A/B，transformers SDPA 三次中位数为 1.409 秒，独立 FlashAttention 2.8.3 为 1.435 秒；共驻首轮也没有优势。该 wheel 还会覆盖 vLLM 0.14 的 vendored 导入路径并触发 CUTLASS API 冲突。因此它被明确排除在项目依赖之外；transformers 回退路径使用 SDPA，vLLM 使用其自带内核。

[Qwen-Audio-3.0-ASR 技术报告](https://arxiv.org/abs/2609.07549) 展示了更新的 MoE、热词和专用 streaming 版本；截至测试日期，本项目未找到适合本机离线部署的官方开放权重，因此不采用云 API 替换本地隐私链路。

## 真实云端与浏览器联调

2026-09-18 晚间在正式后台服务上，以真实 DeepSeek `deepseek-flash`、真实 GPU ASR/TTS 和 Chromium AudioWorklet 测得：

| 轮次 | 端点后 ASR | LLM 首 token 增量 | TTS 首 PCM 增量 | 端点至浏览器首播 |
| --- | ---: | ---: | ---: | ---: |
| 第一轮 | 885.2 ms | 460.1 ms | 355.8 ms | 1830.5 ms |
| 第二轮 | 591.5 ms | 467.3 ms | 260.7 ms | 1401.7 ms |
| 播放中打断后恢复 | 963.4 ms | 411.0 ms | 388.2 ms | 1883.9 ms |

输入均为“你好，请用一句话介绍你自己。”，连续两轮准确转写；实际回复为云端生成，不是 loopback 固定答案。播放中按钮打断到收到 clear 约 39.3ms，断线重连后完整恢复。上述首播不包含 280ms VAD 静音判定，也不代表物理声卡延迟。测试只覆盖少量固定句子，不是 P95 或长期稳定性承诺。

浏览器使用虚拟麦克风与虚拟输出，实际执行重采样、WebSocket 音频传输和渲染确认；物理扬声器及音色仍需人工验收。直接 WSLg/PulseAudio 双向流压测发生系统服务阻塞，默认部署因此使用 Windows 浏览器，而不是将该失败隐藏在模型指标中。报告和录音位于 Git 忽略的 `artifacts/verification/browser-report.json`、`browser-response.wav`。

后台 SIGTERM 正常退出，vLLM/原生 TTS/HTTP 线程完成清理；同一机器显存从约 6.4GB 回落至约 1.2GB 的桌面基线。再次启动后重跑浏览器测试通过，重复启动被文件锁拦截。

## 口径

- TTFA/首包：调用开始至引擎返回第一块非空 PCM。
- 可播放首包：包含 TTS startup buffer 后进入应用播放队列的时间。
- RTF：合成墙钟时间 / 输出音频时长，低于 1 才能持续实时播放。
- GPU 数字包含各运行时 CUDA context，不能直接与 `torch.cuda.max_memory_allocated()` 混用。

后续质量门槛应加入固定的普通话、数字/英文混读、方言和噪声语料，计算 ASR CER，并对 TTS 做至少 ABX 音色相似度和自然度盲测。
