from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
prompt=PromptTemplate(
    template="Generate a detaliled report on {topic}",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="Generate a  5 pointer  summary from the following text\n {text}",
    input_variables=["text"],
)
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

parser = StrOutputParser()
chain= prompt | model | parser | prompt2 | model | parser

result=chain.invoke({
    "topic":"Climate change"})
print(result)
