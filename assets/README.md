# 本地参考音频

将有权使用的单人干净录音放为 `assets/ref_audio.wav`，或通过 `TTS_REF_AUDIO` 指定其他本地路径。建议从约 3 秒、无背景音乐的语音开始；音色与自然度需要实际试听。

`TTS_REF_TEXT` 留空时使用 speaker embedding；填写准确转录文本时使用完整 ICL 克隆。不要填写与音频不符的占位文本。

WAV 文件不进入 Git，新 clone 必须自行准备参考音频。模型权重使用 Hugging Face 缓存，不放在此目录；测试生成的 WAV 放入 `artifacts/verification/`。
