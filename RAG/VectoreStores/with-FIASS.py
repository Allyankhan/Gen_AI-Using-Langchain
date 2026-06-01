from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Babar Azam is one of the most successful cricketers in the Pakistan Super League. Known for his elegant cover drive.",
    metadata={"team": "Peshawar Zalmi"}
)

doc2 = Document(
    page_content="Shaheen Afridi is a fast bowler known for his deadly yorkers and swing bowling in international cricket.",
    metadata={"team": "Lahore Qalandars"}
)

doc3 = Document(
    page_content="Shadab Khan is an all-rounder who contributes with both leg-spin bowling and aggressive batting in T20 matches.",
    metadata={"team": "Islamabad United"}
)

doc4 = Document(
    page_content="Mohammad Rizwan is a consistent wicketkeeper-batsman known for his fitness and match-winning performances.",
    metadata={"team": "Multan Sultans"}
)

doc5 = Document(
    page_content="Fakhar Zaman is an explosive opening batsman famous for his attacking style in limited-overs cricket.",
    metadata={"team": "Lahore Qalandars"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

# embeddings
embeddings = OpenAIEmbeddings()

# create vector store
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

# similarity search
results = vector_store.similarity_search(
    query="who among these are a bowler?",
    k=2
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 50)