from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from langchain_core.runnables import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableBranch,
    RunnableLambda
)

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize this report in one line:\n{report}",
    input_variables=["report"]
)

parser = StrOutputParser()

# Report generator
report_gen_chain = RunnableSequence(
    prompt,
    model,
    parser
)

# Branch
branch_chain = RunnableBranch(

    (
        lambda x: len(x.split()) > 10,

        RunnableSequence(
            RunnableLambda(lambda x: {"report": x}),
            prompt2,
            model,
            parser
        )
    ),

    RunnablePassthrough()
)

# Final chain
final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain
)

result = final_chain.invoke({
    "topic": "Russia Ukraine war"
})

print(result)