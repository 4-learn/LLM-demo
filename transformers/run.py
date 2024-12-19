from transformers import pipeline

# 建立一個文本生成器（使用 GPT-2 模型）
generator = pipeline("text-generation", model="gpt2")

# 生成文本時啟用截斷
prompt = "Once upon a time, in a distant land"
result = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)

print(str(type(result)))

# 顯示生成的文本
print(str(result))
print(result[0]['generated_text'])
