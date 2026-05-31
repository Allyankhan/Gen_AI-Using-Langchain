from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
WordPress is an open source Content Management System (CMS), which allows the users to build 
dynamic websites and blogs. WordPress is the most popular blogging system on the web and allows 
updating, customizing and managing the website from its back-end CMS and components.  
The Content Management System (CMS) is a software which stores all the data such as text, photos, 
music, documents, etc. and is made available on your website. 

"""

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)
print(chunks)