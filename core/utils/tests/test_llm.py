from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 模型 ID
MODEL_ID = "Qwen/Qwen3-1.7B-FP8"

def test_llm():
    print(f"Loading LLM model: {MODEL_ID} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # 尝试使用 auto dtype，如果显存足够且库支持，它会自动选择最佳精度
        # 注意：对于 FP8 模型，如果 GPU 不支持 FP8，可能需要手动指定 torch_dtype=torch.float16
        # 如果加载失败，可以尝试把 torch_dtype 改为 torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype="auto", 
            device_map="auto"
        )
        print("Model loaded successfully.")

        prompt = "Give me a short introduction to AI."
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print("Generating response...")
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100
        )
        
        # 解码输出
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("--- LLM Result ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("------------------")
        print("✅ LLM Test Passed!")

    except Exception as e:
        print(f"❌ LLM Test Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm()
