from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

# Create Model
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Prompt Template
prompt = PromptTemplate(
    template="What year did {company} go public?",
    input_variables=["company"]
)

# Output Parser
parser = StrOutputParser()

# Create Chain
chain = prompt | model | parser

# User Input
topic = input("Enter a company name: ")

# Run Chain
response = chain.invoke({
    "company": topic
})

# Print Output
print("\nResponse:")
print(response)