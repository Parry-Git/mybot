import sys
import os

# Ensure the 'bot' directory is in PYTHONPATH so we can import 'bot.core...'
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from bot.core.llm.llm_client import OpenAILLMClient

def test_deepseek_api():
    print("Testing DeepSeek API Streaming...")
    try:
        client = OpenAILLMClient()
    except ValueError as e:
        print(f"Failed to initialize client: {e}")
        return

    history = [
        {"role": "user", "content": "Hello! Give me a short, 2-sentence introduction to AI."}
    ]
    
    print(f"User: {history[0]['content']}")
    print("Assistant: ", end="", flush=True)
    
    try:
        generator = client.stream_chat(history)
        for token in generator:
            print(token, end="", flush=True)
        print("\n\n--- End of Stream ---")
        print("✅ DeepSeek API Streaming Test Passed!")
    except Exception as e:
        print(f"\n❌ DeepSeek API Streaming Test Failed: {e}")

if __name__ == "__main__":
    test_deepseek_api()
