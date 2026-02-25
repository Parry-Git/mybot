import torch
import psutil
import os
import time
import GPUtil

def print_monitor():
    print("\033c", end="") # Clear terminal
    print("=== MyBot Real-time Resource Monitor ===")
    
    # GPU Monitor
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        print(f"  Memory: {gpu.memoryUsed}MiB / {gpu.memoryTotal}MiB ({gpu.memoryUtil*100:.1f}%)")
        print(f"  Load:   {gpu.load*100:.1f}%")
        print(f"  Temp:   {gpu.temperature}C")
    
    # System RAM Monitor
    ram = psutil.virtual_memory()
    print(f"
System RAM:")
    print(f"  Used:   {ram.used / (1024**3):.2f} GB / {ram.total / (1024**3):.2f} GB ({ram.percent}%)")
    
    # Python Process Monitor
    process = psutil.Process(os.getpid())
    print(f"
This Process Memory: {process.memory_info().rss / (1024**3):.2f} GB")
    
    if torch.cuda.is_available():
        print(f"
Torch CUDA Memory Summary:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MiB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / (1024**2):.2f} MiB")

if __name__ == "__main__":
    while True:
        try:
            print_monitor()
            time.sleep(1)
        except KeyboardInterrupt:
            break
