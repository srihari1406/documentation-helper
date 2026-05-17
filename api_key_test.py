from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

load_dotenv()

MODEL = "gpt-3.5-turbo"
llm = init_chat_model(f"openai:{MODEL}")

result = llm.invoke("What is the capital of France?")
print(result.content)
