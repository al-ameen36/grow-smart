from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from prompt_templates import disease_template, chat_template
from agent import query_engine
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

load_dotenv()

Settings.llm = OpenAI(temperature=0.2, model="gpt-4o")
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    q: str


@app.post("/chat")
async def ask(query: Query):
    formatted_query = chat_template.format(user_query=query.q)
    response = query_engine.query(formatted_query)
    return {"response": response.response}


@app.post("/disease")
async def ask(query: Query):
    formatted_query = disease_template.format(user_query=query.q)
    response = query_engine.query(formatted_query)
    return {"response": response.response}
