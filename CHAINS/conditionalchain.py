from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser
)

from langchain_core.runnables import RunnableBranch

# Load environment variables
load_dotenv()

# -------------------------
# Model
# -------------------------
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9
)

# -------------------------
# Pydantic Schema
# -------------------------
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the feedback"
    )

# -------------------------
# Parser
# -------------------------
parser = PydanticOutputParser(pydantic_object=Feedback)

# -------------------------
# Classification Prompt
# -------------------------
prompt_classifier = PromptTemplate(
    template="""
Classify the following feedback into positive, negative, or neutral.

{format_instructions}

Feedback: {feedback}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

classifier_chain = prompt_classifier | model | parser

# -------------------------
# Positive Response Chain
# -------------------------
positive_prompt = PromptTemplate(
    template="""
Generate a friendly thank-you response for this positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)

positive_chain = positive_prompt | model | StrOutputParser()

# -------------------------
# Negative Response Chain
# -------------------------
negative_prompt = PromptTemplate(
    template="""
Generate an apology and improvement response for this negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)

negative_chain = negative_prompt | model | StrOutputParser()

# -------------------------
# Neutral Response Chain
# -------------------------
neutral_prompt = PromptTemplate(
    template="""
Generate a professional acknowledgment response for this neutral feedback:

{feedback}
""",
    input_variables=["feedback"]
)

neutral_chain = neutral_prompt | model | StrOutputParser()

# -------------------------
# Runnable Branch (FIXED)
# -------------------------
branch_chain = RunnableBranch(
    (
        lambda x: x["sentiment"] == "positive",
        positive_chain
    ),
    (
        lambda x: x["sentiment"] == "negative",
        negative_chain
    ),
    (
        lambda x: x["sentiment"] == "neutral",
        neutral_chain
    ),
    neutral_chain
)

# -------------------------
# Input
# -------------------------
feedback_text = "The product is amazing and works perfectly!"

# -------------------------
# Step 1: Classify sentiment
# -------------------------
classification = classifier_chain.invoke({
    "feedback": feedback_text
})

print("Classification Result:")
print(classification)

# -------------------------
# Step 2: Prepare input for branching
# -------------------------
branch_input = {
    "feedback": feedback_text,
    "sentiment": classification.sentiment
}

# -------------------------
# Step 3: Run branch
# -------------------------
result = branch_chain.invoke(branch_input)

print("\nFinal Response:")
print(result)