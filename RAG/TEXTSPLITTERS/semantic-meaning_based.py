from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text = """
LangChain helps build AI applications.
It supports RAG systems and agents.

Python is a popular programming language.
It is widely used in machine learning.

FAISS and Chroma are vector databases.
They store embeddings efficiently.
"""

embeddings = OpenAIEmbeddings()

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i}:\n{chunk}")