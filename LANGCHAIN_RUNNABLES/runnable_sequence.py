from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

# Load API key
load_dotenv()

model= ChatOpenAI()

prompt=PromptTemplate(
    template="write a joke about {topic}",
    input_variables={'topic'}
)

parser=StrOutputParser()

prompt2=PromptTemplate(
    template="Explain the following  {joke}",
    input_variables={'joke'}
)

chain=RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))
