from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)
documents=[
    "Islamabad is the capital of Pakistan.",
    "Karachi is the largest city of Pakistan.",
    "Lahore is the cultural capital of Pakistan."
]
result = embedding.embed_documents(documents)

print(str(result))