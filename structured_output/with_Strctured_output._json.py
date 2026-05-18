from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Proper JSON Schema
json_schema = {
    "title": "MobilePhone",
    "description": "Information about a mobile phone",
    "type": "object",
    "properties": {
        "brand": {
            "type": "string",
            "description": "Brand of the phone"
        },
        "model": {
            "type": "string",
            "description": "Model name"
        },
        "price": {
            "type": "integer",
            "description": "Price in dollars"
        },
        "features": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of phone features"
        }
    },
    "required": ["brand", "model", "price"]
}

structured_model = model.with_structured_output(json_schema)

response = structured_model.invoke(
    "Tell me about the iPhone 15 Pro"
)

print(response) 