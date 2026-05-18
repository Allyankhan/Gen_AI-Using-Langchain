from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant that answers questions about the world.
    
    Question: {question}
    Answer:
    """,
    input_variables=["question"],
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the capital of Pakistan?"
})

# print(result)
chain.get_graph().print_ascii()#visulaize the chain structure

