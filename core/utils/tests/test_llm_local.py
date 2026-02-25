import sys
import os

# Ensure the 'bot' directory is in PYTHONPATH so we can import 'bot.core...'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from bot.core.llm.llm_client import LocalLLMClient
from bot.core.llm.streamer import TokenAggregator

def test_local_llm_with_aggregator():
    print("Testing Local LLM Streaming with Token Aggregator (Qwen3-1.7B-FP8)...")
    try:
        client = LocalLLMClient()
    except Exception as e:
        print(f"Failed to initialize LocalLLMClient: {e}")
        return

    history = [
        {"role": "user", "content": "你好！请用两句话简短地介绍一下你自己。"}
    ]
    
    print(f"\nUser: {history[0]['content']}")
    print("Assistant:")
    
    aggregator = TokenAggregator()
    
    try:
        generator = client.stream_chat(history)
        chunk_generator = aggregator.aggregate(generator)
        
        for i, chunk in enumerate(chunk_generator):
            print(f"  [Chunk {i+1}]: {chunk}")
        print("\n--- End of Stream ---")
        print("✅ Local LLM Streaming Test Passed!")
    except Exception as e:
        print(f"\n❌ Local LLM Streaming Test Failed: {e}")

if __name__ == "__main__":
    test_local_llm_with_aggregator()
