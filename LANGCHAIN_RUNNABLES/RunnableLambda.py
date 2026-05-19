from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel,RunnableLambda

# Load API key
load_dotenv()

def word_count(text):
    return len(text.split())

model= ChatOpenAI()

prompt=PromptTemplate(
    template="write a joke about {topic}",
    input_variables={'topic'}
)

parser=StrOutputParser()
joke_gen_Chain=RunnableSequence(prompt, model, parser)


parallel_chain=RunnableParallel(
    {
        "JOKE":RunnablePassthrough(),
        "explanation":RunnableSequence(word_count, RunnableLambda(lambda x: f"The joke has {x} words"), parser)
    }
)

final_chain=RunnableSequence(joke_gen_Chain, parallel_chain)
print(final_chain.invoke({'topic':'cricket'}))



