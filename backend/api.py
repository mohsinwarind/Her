# from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
load_dotenv() #loading dot env file
# client = OpenAI(api_key=os.getenv("openAi_KEY"))

# response = client.responses.create(
#     model="gpt-4o-mini",
#     input="Write a one-sentence bedtime story about a unicorn.",
# )

# print(response.output_text)

client = InferenceClient(token=os.getenv("HF_TOKEN"))
selected_model = "openai/gpt-oss-120b"
response = client.chat_completion(
    messages=[{"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}],
    model=selected_model,
    max_tokens=100,
)
print(response.choices[0].message.content)