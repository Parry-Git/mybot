Project 🤖 Mybot

1. 项目目标
    * 全pipeline流式的低延迟语音机器人
    * 技术选型：asr + llm + tts
    * asr：model/Qwen/Qwen3-ASR-0.6B 
    * tts：model/Qwen/Qwen3-TTS-12Hz-0.6B-Base 
    * llm：**双线并行**
        * 方案 A (完全体): 本地 Qwen3-1.7B-FP8 (追求极致隐私与全栈掌控)
        * 方案 B (实用派): 在线大模型 API (deepseek/google gemini格式，追求高智能与低显存)
    *   推理后端：暂时选用纯transformers后端，后续考虑换用vllm加速
    *   [详细架构设计与开发路线](./architecture.md)
        
2. 项目架构
├── bot    # 核心开发文件夹
│   ├── core
│   │   ├── asr
│   │   │   ├── microphone.py
│   │   │   └── vad.py
│   │   ├── llm
│   │   │   ├── llm_client.py    # LLM 统一接口与实装 (API & 本地)
│   │   │   └── streamer.py      # 流式断句与 Token 聚合 (兼容 <think> 标签过滤)
│   │   ├── tts
│   │   └── utils    # 工具与测试等
│   │       ├── config.py    # 统一参数管理
│   │       ├── tests    # 测试文件统一收纳
│   │       └── tools    # 工具函数与文件统一收纳
│   ├── docs    # 文档与开发进度
│   │   ├── architecture.md
│   │   ├── overview.md
│   │   └── pipeline.md
│   ├── README.md    # 环境配置相关，已经测试跑通
│   └── requirements.txt    # 可等待后续开发中补充相关依赖
└── reference    # 用于即时查看官方文档、examples以及部分接口的实现
    ├── Qwen3
    │   └── ......
    ├── Qwen3-ASR
    │   └── ......
    ├── Qwen3-TTS
    │   └── ......
    └── Qwen3-VL
        └── ......

> 注：reference下的参考仓库不参与开发，项目中的qwen-tts与qwen-asr通过直接pip install安装而不是pip install -e .这种方式，希望简化项目的配置流程与开发，放置在reference是用于实时查看与参考，帮助开发
> 但是这非常重要，一切与模型相关的代码编写前都请阅读并参考官方代码库中的文档和示例代码


3. 模型选择

对应于项目目标，当前缓存了如下模型
```
hf cache scan
REPO ID               REPO TYPE SIZE ON DISK NB FILES REFS LOCAL PATH                                                                    
----------------------------------------------------------------------------------------------------- 
Qwen/Qwen3-1.7B-FP8           model     2.7G     main /home/parry-wsl/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-FP8           
Qwen/Qwen3-ASR-0.6B           model     1.9G     main /home/parry-wsl/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B           
Qwen/Qwen3-TTS-12Hz-0.6B-Base model     2.5G     main /home/parry-wsl/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base 
Qwen/Qwen3-VL-2B-Instruct-FP8 model     3.5G     main /home/parry-wsl/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct-FP8 
```
（注：其中qwen-vl等待后续版本启动模型视觉输入相关开发时备用，暂时可忽略）

4. 本地硬件配置
处理器：AMD Ryzen 9 7940H
内存：16G DDR5 4800MHz
显卡：NVIDIA GeForce RTX 4060 Laptop GPU 8GB （独立显卡）/ AMD Radeon 780M Graphics （集成显卡）
开发环境位于wsl ubuntu

5. 关键技术决策记录 (2026-02-10 & 2026-02-25)
*   **硬件可行性验证**: 
    *   经实测，在 RTX 4060 (8GB) 上同时加载 ASR(FP16) + TTS(FP16) + LLM(FP8) 是可行的。
    *   **显存水位**: Allocated ~6GB / Reserved ~7.2GB。处于"可用但紧凑"状态，剩余约 1GB 动态空间用于 KV Cache 和推理中间状态。
*   **量化策略**: 
    *   **ASR & TTS**: 强制保持 **FP16/BF16**。因 0.6B 模型量化收益较低，但对语音质量/识别率影响巨大，不建议量化。
    *   **LLM**: 采用 **FP8** 或更激进的 **Int4** 以换取显存空间，或换用在线大模型api
*   **开发路线修正与进度**: 
    *   LLM 模块已采用 **Interface 接口化设计**，同时支持了 "本地模型 (LocalLLMClient)" 和 "在线 API (OpenAILLMClient)"。
    *   目前在线 API 方案（以 DeepSeek API 为例）的流式调用 (`stream_chat`) 与 `TokenAggregator` (语段级断句并过滤 `<think>` 标签) 已**全部跑通并完成适配**。
    *   **关于本地模型优化:** 经测试，纯 `transformers` 的 1.7B 模型生成速度不如 GGUF/llama.cpp 工具的预期，存在性能瓶颈。
    *   **当前结论:** 决定暂时以**在线 API (方案 B)** 为主线，继续开发后续流程。本地模型在后续阶段再考虑引入 vLLM 或 llama.cpp 等加速方案进行针对性优化。

6. 开发进度跟踪 (Development Progress)

*   [x] **Phase 1: 基础听觉构建 (The Ear)**
    *   [x] Microphone 录音封装
    *   [x] VAD 语音活动检测
    *   [x] 集成测试 (wav 保存)
*   [x] **Phase 2: 感知与表达 (Perception & Expression)**
    *   [x] ASR Client (Qwen3-ASR)
    *   [x] TTS Client (Qwen3-TTS)
    *   [x] Audio Player (实时播放)
    *   [x] 集成测试 (Echo Bot 复读机)
*   [x] **Phase 3: 大脑接入与流式优化 (The Brain & Streaming)**
    *   [x] LLM Interface 定义
    *   [x] OpenAI Client (API 模式) 及流式适配
    *   [x] Token Aggregator (流式断句聚合与过滤)
    *   [x] Chat Bot 集成 (全流程)
*   [ ] **Phase 4: 系统完善 (Polishing)**
    *   [ ] 打断机制
    *   [ ] 提示音效
    *   [ ] 长期记忆 (Optional)

7. 开发注意事项 (Development Notes)

*   **关于 Gemini CLI 与 Python 字符串转义的重要提示**:
    *   **问题描述**: 在使用 Gemini CLI 的 `write_file` 或 `replace` 工具向 Python 文件中写入包含换行符 `\n` 的字符串时，可能会遇到 `SyntaxError: untermined string literal` 的错误。
    *   **原因分析**: 这是因为 Gemini CLI 在生成工具调用的 JSON 负载时，会将字符串中的 `\n` 直接解释为物理换行，而不是保留为 `\` 和 `n` 两个字符。这导致写入文件中的 Python 字符串字面量被破坏。
    *   **解决方案与最佳实践**: 当你需要通过 Gemini CLI 在 Python 代码中写入包含 `\n` 等特殊转义字符的字符串时，请务必在你的指令中，或者在 Gemini CLI 生成代码后，手动确认 `\n` 被正确地表示为字面上的 `\n` 而不是一个实际的换行。如果发现生成的代码存在语法问题，应立即指出并要求其修正，以确保写入文件的内容是语法正确的。
