from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
# Load API key
load_dotenv()

prompt1=PromptTemplate(
    template="Write a tweet about {topic}",
    input_variables={'topic'}
)

prompt2=PromptTemplate(
    template="Write a LinkedIn post about {topic}",
    input_variables={'topic'}
)
model= ChatOpenAI()

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
        "tweet":RunnableSequence(prompt1, model, parser),
        "linkedin":RunnableSequence(prompt2, model, parser)
    }
)

print(parallel_chain.invoke({'topic':'AI'}))
