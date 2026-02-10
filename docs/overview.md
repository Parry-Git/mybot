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
│   │   ├── tts
│   │   └── utils    # 工具与测试等
│   │       ├── config.py
│   │       ├── tests
│   │       └── tools
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


1. 模型选择

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

5. 关键技术决策记录 (2026-02-10)
*   **硬件可行性验证**: 
    *   经实测，在 RTX 4060 (8GB) 上同时加载 ASR(FP16) + TTS(FP16) + LLM(FP8) 是可行的。
    *   **显存水位**: Allocated ~6GB / Reserved ~7.2GB。处于"可用但紧凑"状态，剩余约 1GB 动态空间用于 KV Cache 和推理中间状态。
*   **量化策略**: 
    *   **ASR & TTS**: 强制保持 **FP16/BF16**。因 0.6B 模型量化收益较低，但对语音质量/识别率影响巨大，不建议量化。
    *   **LLM**: 采用 **FP8** 或更激进的 **Int4** 以换取显存空间，或换用在线大模型api
*   **开发路线修正**: 
    *   LLM 模块将采用 **Interface 接口化设计**，同时支持 "本地模型" 和 "在线 API"。
    *   优先确保 API 模式跑通全流程，本地模型作为进阶目标，需配合严格的 Context Sliding Window (滑动窗口) 机制防止 OOM。
