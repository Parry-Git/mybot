import sys
import os

# Ensure the 'bot' directory is in PYTHONPATH so we can import 'bot.core...'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from bot.core.llm.llm_client import OpenAILLMClient
from bot.core.llm.streamer import TokenAggregator

def test_deepseek_api_with_aggregator():
    print("Testing DeepSeek API Streaming with Token Aggregator...")
    try:
        client = OpenAILLMClient()
    except ValueError as e:
        print(f"Failed to initialize client: {e}")
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
        print("✅ DeepSeek API Streaming Test with Aggregator Passed!")
    except Exception as e:
        print(f"\n❌ DeepSeek API Streaming Test with Aggregator Failed: {e}")

if __name__ == "__main__":
    test_deepseek_api_with_aggregator()
