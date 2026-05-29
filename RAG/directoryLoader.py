from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

loader = DirectoryLoader(
    path=r"D:\Machine Learning\Gen_AI\Langchain\RAG",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
    silent_errors=True
)
# docs=loader.load()
docs = loader.lazy_load()
for document in docs:
     print(document.metadata)

  

# print("Documents Loaded:", len(docs))