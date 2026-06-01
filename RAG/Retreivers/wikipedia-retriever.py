from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
load_dotenv()

retriever=WikipediaRetriever(top_k_results=2, lang="en")

query="The geopolitics of india and pakistan "

docs= retriever.invoke(query)



