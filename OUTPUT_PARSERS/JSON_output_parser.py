from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

# Create parser
parser = JsonOutputParser()

template = PromptTemplate(
    template="""
Give me the name, age, and occupation of a person based on the following description.

{format_instructions}

Description:
{description}
""",
    input_variables=["description"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

prompt = template.format(
    description="A 25 year old software engineer who loves AI and works at a tech company."
)

chain=template | model | parser
result=chain.invoke(prompt)
print(result)