import torch
from qwen_asr import Qwen3ASRModel

# 模型 ID
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
# 测试音频 URL (英语)
AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"

def test_asr():
    print(f"Loading ASR model: {MODEL_ID} ...")
    try:
        # 加载模型
        model = Qwen3ASRModel.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        print("Model loaded successfully.")

        print(f"Transcribing audio from: {AUDIO_URL}")
        # 推理
        results = model.transcribe(
            audio=AUDIO_URL,
            language=None, # 自动检测
        )
        
        print("--- ASR Result ---")
        print(f"Language: {results[0].language}")
        print(f"Text: {results[0].text}")
        print("------------------")
        print("✅ ASR Test Passed!")

    except Exception as e:
        print(f"❌ ASR Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_asr()
