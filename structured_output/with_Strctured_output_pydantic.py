from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9
)

# Schema
class Review(BaseModel):
    review: str = Field(description="The review text")
    rating: int = Field(description="The rating between 1 and 5")
    pros: list[str] = Field(description="List of pros")
    cons: list[str] = Field(description="List of cons")
# Structured output model
structured_model = model.with_structured_output(Review)

# Invoke model
response = structured_model.invoke(
    "What do you think about the movie Inception?"
) 

print(response)