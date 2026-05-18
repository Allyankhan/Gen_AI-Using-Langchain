from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",   # v0.2 works much better
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

result = llm.invoke("What is the capital of Pakistan?")
print(result)