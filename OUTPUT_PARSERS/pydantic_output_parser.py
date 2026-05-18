from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

# Define schema
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    occupation: str = Field(description="The person's occupation")

# Convert model into structured output model
structured_model = model.with_structured_output(Person)

template = PromptTemplate(
    template="""
Give me the name, age, and occupation of a person based on this description:

{description}
""",
    input_variables=["description"]
)

chain = template | structured_model

result = chain.invoke({
    "description": "A 25 year old software engineer who loves AI and works at a tech company."
})

print(result)