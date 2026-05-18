from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# OpenAI Model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)


model2=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)


# Prompt for Notes
prompt_notes = PromptTemplate(
    template="""
Generate short and simple notes from the following text:

{text}
""",
    input_variables=["text"]
)

# Prompt for Quiz
prompt_quiz = PromptTemplate(
    template="""
Generate 5 question answers from the following text:

{text}
""",
    input_variables=["text"]
)

# Prompt for Merging
prompt_merge = PromptTemplate(
    template="""
Merge the following notes and quiz into one well-formatted document.

NOTES:
{notes}

QUIZ:
{quiz}
""",
    input_variables=["notes", "quiz"]
)

# Output Parser
parser = StrOutputParser()

# Parallel Chain
parallel_chain = RunnableParallel(
    {
        "notes": prompt_notes | model | parser,
        "quiz": prompt_quiz | model2 | parser
    }
)

# Merge Chain
merge_chain = prompt_merge | model | parser

# Final Chain
chain = parallel_chain | merge_chain

# Input Text
text = """
Climate change is a long-term alteration in Earth's climate,
primarily due to human activities such as burning fossil fuels
and deforestation. It leads to rising global temperatures,
melting ice caps, and more frequent extreme weather events.
Addressing climate change requires reducing greenhouse gas
emissions and transitioning to renewable energy sources.
"""

# Run Chain
result = chain.invoke({
    "text": text
})

# Print Result
print(result)

# Visualize Chain Structure
chain.get_graph().print_ascii()