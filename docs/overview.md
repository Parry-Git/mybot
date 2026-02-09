Project 🤖 Mybot

1. 项目目标
    * 全pipeline流式的低延迟语音机器人
    * 技术选型：asr + llm + tts
    * asr：model/Qwen/Qwen3-ASR-0.6B 
    * tts：model/Qwen/Qwen3-TTS-12Hz-0.6B-Base 
    * llm：model/Qwen/Qwen3-1.7B-FP8 / model/Qwen/Qwen3-VL-2B-Instruct-FP8 (待后续开发) / 在线大模型api
    * 推理后端：暂时选用纯transformers后端，后续考虑换用vllm加速

2. 项目架构
.
├── bot    # 核心开发文件夹
│   ├── core
│   │   ├── asr
│   │   ├── llm
│   │   ├── tts
│   │   └── utils    # 工具与测试等
│   ├── docs    # 文档与开发进度
│   │   └── overview.md
│   └── requirements.txt
└── reference
    ├── Qwen3
    │   └── ......
    ├── Qwen3-ASR
    │   └── ......
    ├── Qwen3-TTS
    │   └── ......
    └── Qwen3-VL
        └── ......