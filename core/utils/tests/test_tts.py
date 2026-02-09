import torch
import soundfile as sf
import os
import requests
from qwen_tts import Qwen3TTSModel

# 模型 ID
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# 复用 ASR 测试中有效的音频 URL (真实人声)
REAL_REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
LOCAL_REF_AUDIO = "ref_audio_downloaded.wav"
# 这段音频的内容大概是 (根据 ASR 测试结果填写，或者写个通用的)
REF_TEXT = "Mister Quilter is the apostle of the middle classes and we are glad to welcome his gospel." 

# 想要生成的文本
TARGET_TEXT = "Hello! This is a test for Qwen3 TTS system. I am running on a local machine using a real reference audio."

OUTPUT_FILE = "output_tts_test.wav"

def download_audio(url, filename):
    """下载音频文件到本地"""
    print(f"Downloading reference audio from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved to: {filename}")
    else:
        raise Exception(f"Failed to download audio. Status code: {response.status_code}")

def test_tts():
    # 1. 下载真实参考音频
    if not os.path.exists(LOCAL_REF_AUDIO):
        try:
            download_audio(REAL_REF_AUDIO_URL, LOCAL_REF_AUDIO)
        except Exception as e:
            print(f"Download failed: {e}. Falling back to dummy generation logic if needed (not implemented here).")
            return

    # 转换为绝对路径
    ref_audio_path = os.path.abspath(LOCAL_REF_AUDIO)

    print(f"Loading TTS model: {MODEL_ID} ...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2", 
        )
        print("Model loaded successfully.")

        print("Generating audio (Voice Cloning)...")
        wavs, sr = model.generate_voice_clone(
            text=TARGET_TEXT,
            language="English",
            ref_audio=ref_audio_path,
            ref_text=REF_TEXT,
        )

        # 保存结果
        sf.write(OUTPUT_FILE, wavs[0], sr)
        
        print(f"\n--- TTS Result ---")
        print(f"Audio generated and saved to: {os.path.abspath(OUTPUT_FILE)}")
        print("------------------\n")
        print("✅ TTS Test Passed!")

    except Exception as e:
        print(f"\n❌ TTS Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tts()