from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Pakistan?"),
]
result = model.invoke(messages)
print(result.content)
messages.append(AIMessage(content=result.content))
print(messages)