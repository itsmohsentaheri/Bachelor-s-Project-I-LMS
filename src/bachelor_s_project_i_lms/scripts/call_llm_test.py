
import os
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("LLM_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Hello! Are you working?"}
        ]
    )

    print(response.choices[0].message.content)

except APIStatusError as e:
    print("Status Code:", e.status_code)
    print("Error Details:", e.response.text)

except Exception as e:
    print("Error:", str(e))
