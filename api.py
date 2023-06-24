from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv

load_dotenv()
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

db = Chroma(
    persist_directory="./langchain_webinar_db", embedding_function=OpenAIEmbeddings()
)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo-0613"),
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True,
)

from pydantic import BaseModel


class Query(BaseModel):
    message: str


@app.post("/chat")
def read_item(query: Query):
    return qa(query.message)
