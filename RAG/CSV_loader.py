from langchain_community.document_loaders import CSVLoader

loader=CSVLoader("RAG/Titanic-Dataset.csv")

docs=loader.load()

print(len(docs))
print(docs[1])
