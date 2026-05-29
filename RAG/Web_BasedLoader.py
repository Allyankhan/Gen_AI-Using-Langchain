from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url='https://www.daraz.pk/products/infinix-hot-60-pro-8gb128gb-pta-approved-i916122027-s3964580787.html?spm=a2a0e.tm80335142.3349068300.20&&scm=1007.51610.379274.0&pvid=b7688e73-f4c5-446b-a8cb-fccb8b84555c&search=flashsale?search=1&mp=1&c=fs&clickTrackInfo=rs%3A0.024%3Bfs_item_discount_price%3A52399%3Bitem_id%3A916122027%3Bpctr%3A0.0%3Bcalib_pctr%3A0.0%3Bvoucher_price%3A52399%3Bmt%3Amustbuy%3Bpromo_price%3A52399%3Bfs_utdid%3A-1%3Bfs_item_sold_cnt%3A2%3Babid%3A379274%3Bfs_item_price%3A55999%3Bpvid%3Ab7688e73-f4c5-446b-a8cb-fccb8b84555c%3Bfs_min_price_l30d%3A0%3Bdata_type%3Aflashsale%3Bfs_pvid%3Ab7688e73-f4c5-446b-a8cb-fccb8b84555c%3Btime%3A1780063843%3Bfs_biz_type%3Afs%3Bscm%3A1007.51610.379274.%3Bchannel_id%3A0000%3Bfs_item_discount%3A6%25%3Bcampaign_id%3A368944&scm=1007.51610.379274.0'
loader=WebBaseLoader(url)

model=ChatOpenAI()

prompt=PromptTemplate(
    template="Answer the following question \n{question} from the following text",
    input_variables=["question",'text']
)

parser=StrOutputParser()

docs=loader.load()
print(docs[0].page_content)

chain= prompt | model | parser

print(chain.invoke({'question':'what is the peak brightness of this product','text':docs[0].page_content}))