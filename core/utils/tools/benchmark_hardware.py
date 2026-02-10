import time
import torch
import gc
# 必须导入以注册自定义架构
try:
    from qwen_asr import Qwen3ASRModel
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("❌ Critical Error: qwen_asr or qwen_tts not found. Please pip install them.")
    exit(1)

from transformers import AutoModelForCausalLM

def print_gpu_status(step_name):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    usage = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    print(f"[{step_name}]")
    print(f"  Allocated: {usage:.2f} GB")
    print(f"  Reserved : {reserved:.2f} GB")
    print("-" * 30)

def benchmark_loading():
    print("🚀 Starting Hardware Benchmark for Mybot (RTX 4060 8GB)")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print_gpu_status("Baseline (Empty)")

    # 1. Load ASR (FP16)
    print("Loading ASR (Qwen3-ASR-0.6B) in FP16...")
    try:
        # 使用 Qwen3ASRModel 直接加载
        asr = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=torch.float16,
            device_map="cuda",
        )
        print_gpu_status("After ASR Load")
    except Exception as e:
        print(f"❌ Failed to load ASR: {e}")
        return

    # 2. Load TTS (FP16)
    print("Loading TTS (Qwen3-TTS-12Hz-0.6B-Base) in FP16...")
    try:
        # 使用 Qwen3TTSModel 直接加载
        tts = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            dtype=torch.float16,
            device_map="cuda",
        )
        print_gpu_status("After TTS Load")
    except Exception as e:
        print(f"❌ Failed to load TTS: {e}")
        return

    # 3. Load LLM (FP16/FP8)
    # Note: Qwen3-1.7B-FP8 usually needs auto loading. 
    # If explicit FP8 support isn't available in this transformers version, 
    # it might fallback or error. We try standard loading first.
    print("Loading LLM (Qwen3-1.7B-FP8)...")
    try:
        # Try loading. If the model config specifies fp8, transformers might handle it.
        # Otherwise, we might need to rely on 'device_map="cuda"' to handle placement.
        llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B-FP8",
            device_map="cuda",
            trust_remote_code=True
        )
        print_gpu_status("After LLM Load (Final)")
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        return

    print("✅ Success! All models loaded into VRAM.")
    print("Check the 'Reserved' memory above. If it's < 7.5GB, you are safe.")

if __name__ == "__main__":
    benchmark_loading()
