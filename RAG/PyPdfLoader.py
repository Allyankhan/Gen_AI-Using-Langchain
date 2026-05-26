from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("Metoerology Temperatures - Dry Bulb_Web Bulb_Dew Point.pdf")

docs=loader.load()

print(len(docs))
print(docs[1].metadata)