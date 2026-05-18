from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

template1 = PromptTemplate(
    template="Write a 5 line report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Convert this text into bullet points:\n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = (
    template1
    | model
    | parser
    | template2
    | model
    | parser
)

result = chain.invoke({
    "topic": "the benefits of exercise"
})

print(result)