#Used when we have no common text rather code or something like markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

print(add(2, 3))
print(multiply(2, 3))
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=40,
    chunk_overlap=0
    
)

chunks = splitter.split_text(code)

print(len(code))
