# 开发日志 (Develop Logs)

## 2026-02-26: TTS 性能深度优化与延迟复盘

### 1. 核心问题
* **TTS 高延迟**：初始测试中，Base 模型合成一句话耗时约 27 秒，短语“你好”耗时 10.27 秒。
* **音频卡顿**：使用 `pyaudio` 时，由于 ALSA 缓冲区未及时填充，出现严重的 `underrun` 报错，导致播放断断续续。
* **嵌套推理瓶颈**：识别出 Qwen3-TTS 12Hz 的 Nested LM 架构（Talker 12fps + Predictor 31fps）导致 Python 层和 HuggingFace `generate` 开销巨大。

### 2. 已实施的优化尝试

#### A. 模型与算力分配
* **切换模型**：从 `Base` (ICL 模式) 切换至 `CustomVoice` 模型。虽然排除了参考音频编码的延迟，但整体生成速度提升有限，确认瓶颈在于生成循环。
* **强制 GPU 迁移**：发现 `code_predictor` 等子模块可能未完全加载至 GPU。在 `TTSClient` 初始化中添加了递归 `.to(device).to(dtype)`，确保全流程在 RTX 4060 上运行。
* **BF16 精度**：全程启用 `bfloat16` 推理，在保持精度的同时压榨 Tensor Core 性能。

#### B. 推理参数精简 (关键进展)
* **限制生成步数**：将 `max_new_tokens` 从默认的 2048 限制为 256。有效防止了模型因错过停止符而进行的无效生成，这是延迟从 27s 降至 15s 的主因。
* **关闭子预测器采样**：设置 `subtalker_dosample=False`。在嵌套生成中使用 Greedy Search 避免了昂贵的 CPU/GPU 同步，显著提升了循环效率。
* **停用冗余惩罚**：移除了 `repetition_penalty` 等高开销参数。

#### C. 架构与播放器优化
* **播放器重构**：弃用 `pyaudio`，改用 `sounddevice` 并维持长连接 `OutputStream`。实测解决了 ALSA 的一卡一卡问题，播放更加平滑。
* **更激进的流式切分**：调整 `TokenAggregator`，将语段触发阈值从 12 降低至 8，并增加了更多标点符号（，；：）作为即时切割点，减少了第一段语音的“攒词”等待时间。
* **引入编译优化**：尝试使用 `torch.compile` 编译 `code_predictor`，通过 CUDA Graph 减少 Python 嵌套开销。

### 3. 当前状态 (RTX 4060)
* **短语响应**：“你好”生成时间从 **10.27s -> 3.45s**。
* **句子响应**：整句生成时间从 **27s -> 15s**。
* **结论**：RTF (Real Time Factor) 仍大于 1，意味着长文本生成的等待感依然存在。这是由 Qwen3-TTS 12Hz 的 1x31 嵌套架构决定的。

### 4. 下一步规划
* **真正的 Chunk-wise 流式生成**：研究如何跳过官方 Wrapper 直接调用底层 `forward` 循环，实现生成几帧就播放几帧，而不是生成完整语段再播放。
* **vLLM 加速**：如果后续 vLLM 正式支持 Qwen3-TTS 的多层生成，迁移到 vLLM 预计能获得 5-10 倍的性能提升。
