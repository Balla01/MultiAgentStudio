
import os
MISTRAL_API_KEY = '0TD9nsBifR6Lkr1kOag9aikbCBImYfGg'#os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral/mistral-large-latest"
from crewai import Agent, Task, Crew, Process, LLM

llm = LLM(model=MISTRAL_MODEL, api_key=MISTRAL_API_KEY)
response = llm.call("Explain vector databases in simple words.")
print(response)