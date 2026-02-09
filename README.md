``` bash
conda create -n mybot python=3.12 -y
conda activate mybot

pip install torch==2.8.* torchvision torchaudio
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install -r requirements.txt
pip install -U qwen-tts
pip install -U qwen-asr
# pip install -U qwen-asr[vllm]
```