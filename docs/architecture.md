# 架构与配置

## 数据流与职责

```text
run.sh -> scripts/service.py -> core/web/server.py
                                 |
                              core/app.py
                                 |
浏览器 AudioWorklet -> VoiceInput / VAD -> ASR 流式会话
                                             |
                                    ConversationPipeline
                                             |
                              LLM token -> 短句队列 (2 段)
                                             |
                               Native Qwen TTS / CUDA
                                             |
                                原生 PCM 队列 (240ms)
                                             |
                              Playback 接口 / 在途音频 (2s)
                                             |
                              AudioWorklet 播放 -> 渲染 ACK
```

| 模块 | 做什么 | 不做什么 |
| --- | --- | --- |
| `app.py` | 构造、预热、资源回收，连接回调 | 不实现模型推理或网络协议 |
| `orchestrator.py` | 轮次 Future、LLM/TTS 并发、取消、完整历史 | 不打开声卡，不加载模型 |
| `asr/asr_client.py` | Qwen3-ASR；默认 vLLM，transformers 回退 | 不决定对话策略 |
| `asr/vad.py` / `voice_input.py` | 端点检测、pre-roll、流式识别会话生命周期 | 不采集物理麦克风 |
| `llm/llm_client.py` / `streamer.py` | 持久异步 HTTP、过滤 think、语段切分 | 不生成或播放音频 |
| `tts/native.py` / `base.py` | Qwen3-TTS Q8 PCM 流、原生暂停与取消、合成接口 | 不依赖浏览器或声卡 |
| `audio/base.py` | 调度层依赖的 Playback 协议 | 不选择具体输出设备 |
| `audio/browser.py` | PCM 代际编号、渲染 ACK、在途容量 | 不处理 HTTP 连接 |
| `audio/player.py` / `microphone.py` / `pulse.py` | 可选直接设备输入输出 | 不处理 ASR/TTS |
| `web/server.py` / `static/` | 同源单会话 WebSocket、网页、重采样 | 不在异步接收循环里执行模型推理 |
| `config.py` / `types.py` | 配置验证、数据类型、时钟 | 不依赖入口脚本 |

业务代码只依赖 `core` 内部模块，不能反向导入 `main.py`、`scripts` 或测试。前台 CLI 和 Web 服务共用同一个 MyBot 生命周期与调度器，不维护两套推理逻辑。

## 实时约束

1. 浏览器按 20ms 提交 16kHz 单声道 PCM16。VAD 保存 160ms pre-roll，默认静音 280ms 后结束一轮。
2. vLLM ASR 在说话期间处理增量音频，端点后 finalize；transformers 回退在端点后完整转写。
3. LLM 在专用 asyncio 线程复用连接。强标点及时释放，弱标点至少 8 字，无标点最多 42 字切段。
4. 待合成文本最多两个语段，原生 PCM 队列 240ms，播放器在途最多两秒。下游变慢会暂停原生生成，而不是转移到无界缓存。
5. 每轮共享取消事件。打断取消 HTTP 请求、停止 TTS、清空播放；浏览器用代际编号忽略迟到音频，服务端忽略旧 ACK。
6. 默认半双工，回复期间丢弃麦克风回声。启用语音打断仍建议戴耳机；浏览器设备 AEC 的效果不能由代码保证。
7. 完成播放后才提交完整历史和指标；取消或失败轮次不提交历史。断线取消当前轮次，重连清空前一会话。
8. ExitStack 管理部分初始化失败和正常退出；vLLM 使用命名 worker RPC 关闭分布式进程组。

## 主要配置

完整定义在 `core/config.py`，常用模板在 `.env.example`。重启服务后配置才生效。

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 无 | 云端密钥，仅保存在本地环境 |
| `LLM_MODEL` / `LLM_THINKING` | `deepseek-flash` / `false` | 模型 ID、是否启用思考 |
| `ASR_BACKEND` | `vllm` | 可回退为 `transformers` |
| `ASR_VLLM_GPU_MEMORY_UTILIZATION` | `0.46` | 为共驻 TTS 留显存 |
| `ASR_VLLM_MAX_MODEL_LEN` | `2048` | 识别上下文上限 |
| `ASR_STREAM_CHUNK_SECONDS` | `1.0` | 模型在线窗口 |
| `TTS_QUANT` / `TTS_STARTUP_BUFFER_MS` | `Q8_0` / `80` | 权重量化、首包抗抖动缓存 |
| `TTS_REF_AUDIO` / `TTS_REF_TEXT` | `assets/ref_audio.wav` / 空 | 参考音色；有准确文本时启用完整 ICL |
| `TTS_LOCAL_FILES_ONLY` | 配置默认 false，模板 true | 是否只使用本地模型缓存 |
| `VAD_FRAME_MS` / `VAD_END_SILENCE_MS` | `20` / `280` | 浏览器固定 20ms 帧；端点等待 |
| `BARGE_IN` | `false` | 语音打断，不影响页面按钮 |
| `TTS_TEXT_QUEUE_SIZE` | `2` | 待合成语段上限 |
| `AUDIO_PLAYBACK_QUEUE_SECONDS` | `2.0` | 播放在途 PCM 上限 |
| `AUDIO_BACKEND` | `auto` | 仅直接设备模式使用 PulseAudio/PortAudio |

当前只维护 native Qwen TTS 和远端 LLM，不保留没有部署用途的 Kokoro/本地 LLM 实验分支。历史模型比较仍保存在 [基准文档](benchmarks.md)。

## 指标边界

`asr_ms`、`llm_first_token_ms`、`first_phrase_ms`、`tts_first_audio_ms`、`playback_first_audio_ms`、`turn_ms` 从 VAD 判定结束开始累计，不含 280ms 静音等待。

`llm_ttft_ms`、`tts_ttfa_ms`、`output_queue_ms` 是阶段间增量。浏览器首播表示首个渲染确认到达服务端，不等于物理扬声器出声。单句测试不是 P95、CER 或音色 MOS 质量认证。
