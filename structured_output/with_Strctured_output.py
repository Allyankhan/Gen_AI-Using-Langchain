from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9
)

# Schema
class Review(TypedDict):
    review: Annotated[str, "The review text"]
    rating: Annotated[int, "The rating between 1 and 5"]
    pros: Annotated[list[str], "List of pros"]
    cons: Annotated[list[str], "List of cons"]
# Structured output model
structured_model = model.with_structured_output(Review)

# Invoke model
response = structured_model.invoke(
    "What do you think about the movie Inception?"
) 

print(response)