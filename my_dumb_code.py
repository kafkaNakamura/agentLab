import os

# Initialize Pinecone client directly
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
import glob


# Set your API keys - MUST REPLACE THESE
os.environ["OPENAI_API_KEY"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_5LkoDN_QB8yo9XFYW6cV1NJiMUKpcexqvYM2QNAE6C7qfzdXpuZLtAa47fYrrogLrEY9oT"
PINECONE_API_KEY = "pcsk_5LkoDN_QB8yo9XFYW6cV1NJiMUKpcexqvYM2QNAE6C7qfzdXpuZLtAa47fYrrogLrEY9oT"  # Not using env var to be explicit

pc = Pinecone(api_key="pcsk_5LkoDN_QB8yo9XFYW6cV1NJiMUKpcexqvYM2QNAE6C7qfzdXpuZLtAa47fYrrogLrEY9oT")
index_name = "pinecone-rag"
# Delete existing index if it exists
# if index_name in pc.list_indexes().names():
#     pc.delete_index(index_name)

# if not pc.has_index(index_name):
#     pc.create_index_for_model(
#         name=index_name,
#         cloud="aws",
#         region="us-east-1",
#         embed={
#             "model":"llama-text-embed-v2",
#             "field_map":{"text": "chunk_text"}
#         }
#     )
# pc.configure_index(index_name, dimension=1536)
loader = PyPDFLoader("test_rag.pdf")
docs = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
plit_docs = text_splitter.split_documents(docs)



vector_store = PineconeVectorStore.from_documents(
    documents=plit_docs,
    embedding=embeddings,
    index_name=index_name,
)


