import os
import sys
import time
import torch
import torchaudio

# =========================================================================
# 环境准备说明 (Environment Setup Instructions):
# CosyVoice2 依赖独立的仓库和各种定制化的 C++ 算子/依赖。
# 测试前请按照以下步骤准备：
#
# 1. 克隆官方仓库（必须带子模块）:
#    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# 2. 安装依赖:
#    cd CosyVoice
#    pip install -r requirements.txt
#    # 如果 macOS/Linux 安装 pynini 报错，请使用 conda:
#    # conda install -y -c conda-forge pynini
# 3. 设置 PYTHONPATH 环境变量运行此脚本:
#    export PYTHONPATH=/你的路径/CosyVoice:/你的路径/CosyVoice/third_party/Matcha-TTS
# =========================================================================

# 尝试导入 CosyVoice
try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("❌ 错误: 找不到 CosyVoice 模块。")
    print("请确保已克隆 FunAudioLLM/CosyVoice 仓库，并将该仓库及其 third_party/Matcha-TTS 目录添加到了 PYTHONPATH 中。")
    print("示例: export PYTHONPATH=/path/to/CosyVoice:/path/to/CosyVoice/third_party/Matcha-TTS")
    sys.exit(1)

# 模型路径，此脚本会自动尝试下载
MODEL_DIR = "pretrained_models/CosyVoice2-0.5B"

def download_model_if_needed():
    if not os.path.exists(MODEL_DIR):
        print(f"📥 模型目录 {MODEL_DIR} 不存在，正在使用 huggingface_hub 自动下载...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", local_dir=MODEL_DIR)
            print("✅ 模型下载完成！")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("请尝试手动下载: git clone https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B pretrained_models/CosyVoice2-0.5B")
            sys.exit(1)

def test_streaming_and_speed(cosyvoice, text, prompt_text, prompt_audio_16k, test_name=""):
    print(f"\n[{test_name}] 开始测试...")
    print(f"🗣️  测试文本: '{text}'")
    
    start_time = time.time()
    first_chunk_time = None
    total_audio_len = 0
    chunks = 0
    
    # 开启 stream=True 进行流式推理
    # inference_zero_shot 接收: tts_text, prompt_text, prompt_speech_16k
    generator = cosyvoice.inference_zero_shot(text, prompt_text, prompt_audio_16k, stream=True)
    
    try:
        for i, chunk in enumerate(generator):
            chunk_time = time.time()
            if i == 0:
                first_chunk_time = chunk_time - start_time
                print(f"  ⚡ 首包延迟 (Time-To-First-Chunk): {first_chunk_time * 1000:.2f} ms")
            
            audio_data = chunk['tts_speech']
            # audio_data shape 通常为 (1, T)
            audio_length_sec = audio_data.shape[1] / cosyvoice.sample_rate
            total_audio_len += audio_length_sec
            chunks += 1
            # 可以选择在这里把 chunk 保存下来听听音质
            # torchaudio.save(f'test_chunk_{i}.wav', audio_data, cosyvoice.sample_rate)
            
    except Exception as e:
         print(f"  ❌ 流式生成过程中出现错误: {e}")
         return
        
    end_time = time.time()
    total_time = end_time - start_time
    rtf = total_time / total_audio_len if total_audio_len > 0 else 0
    
    print(f"  ✅ 测试完成: 共生成 {chunks} 个音频块，总音频时长: {total_audio_len:.2f} 秒")
    print(f"  ⏱️ 总推理时间: {total_time:.2f} 秒")
    print(f"  🚀 实时率 (RTF - 越小越代表生成速度越快于播放速度): {rtf:.3f}")
    
    if rtf > 1.0:
        print("  ⚠️ 警告: RTF > 1.0，意味着生成速度比真实说话速度慢，播放时可能会卡顿。")
    elif rtf < 0.5:
        print("  🎉 优秀: RTF < 0.5，算力完全足够支撑毫无卡顿的流式对话！")
        
    return first_chunk_time, total_time, rtf

def main():
    download_model_if_needed()
    
    print("\n" + "="*50)
    print("🔄 正在加载 CosyVoice2-0.5B 模型到 GPU (如果可用)...")
    load_start = time.time()
    # 初始化模型，会自动根据 torch.cuda.is_available() 使用 GPU
    cosyvoice = CosyVoice(MODEL_DIR)
    print(f"✅ 模型加载完成，耗时: {time.time() - load_start:.2f} 秒")
    print("="*50 + "\n")

    print("🎙️  准备测试用例数据...")
    # 为了纯测试速度，我们生成一段3秒长的白噪声作为 prompt audio。
    # ⚠️ 警告: 使用随机噪声作为 reference audio 跑出来的音质肯定很差，因为模型学不到真实的音色！
    # 这里纯粹是为了测延迟。如果你后续要测音色和自然度，请传入一段真实的包含人声的 prompt.wav。
    prompt_audio_16k = torch.randn(1, 16000 * 3) # 3秒长，16000采样率
    prompt_text = "这是一段用于克隆音色和提供参考的测试音频。"
    
    print("\n--------------------------------------------------")
    # 用例 1: 纯中文短句 (测试机器人的基础应答首包延迟)
    text_short = "你好，我是你的智能语音助手，现在一切运行正常。"
    test_streaming_and_speed(cosyvoice, text_short, prompt_text, prompt_audio_16k, test_name="用例 1 | 纯中文短句响应测试")
    
    print("\n--------------------------------------------------")
    # 用例 2: 高频中英混合 (测试 Code-switching 能力)
    # 这个句子考验模型是否能在中英文语境间无缝切换，不断句、不结巴、不重载模型。
    text_mixed = "没问题，我刚刚 check 了一下你的 schedule，明天 morning 我们有一个 meeting。需要我帮你 pre-book 一下 conference room 吗？"
    test_streaming_and_speed(cosyvoice, text_mixed, prompt_text, prompt_audio_16k, test_name="用例 2 | 高频中英混合无缝输出测试")
    
    print("\n--------------------------------------------------")
    # 用例 3: 长段落流式持续性 (评估长文本的整体吞吐量和稳定性)
    text_long = "CosyVoice 是一款非常强大的语音生成模型。As we can see, it perfectly handles both English and Chinese contexts smoothly. 此外，它的流式输出能力非常适合用于我们正在开发的语音机器人项目中。If the latency is low enough, this will be a real game changer for our AI voice stack."
    test_streaming_and_speed(cosyvoice, text_long, prompt_text, prompt_audio_16k, test_name="用例 3 | 长段落中英混合持续流式生成测试")

if __name__ == "__main__":
    main()
