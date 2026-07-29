from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv() #loading dot env file
client = OpenAI(api_key=os.getenv("openAi_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a one-sentence bedtime story about a unicorn.",
)

print(response.output_text)