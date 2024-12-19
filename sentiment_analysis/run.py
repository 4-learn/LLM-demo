from transformers import pipeline

# 建立一個情感分析的 pipeline
classifier = pipeline("sentiment-analysis")

# 使用 classifier 分析文本情感
result = classifier("I love using Hugging Face's transformers!")
print(result)
