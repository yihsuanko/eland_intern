from typing import Optional, List, Union
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from app.ner_predict import *
from pydantic import BaseModel, parse_obj_as

TOKEN = "albert_base_chinese_ner_0329"
MODEL = "albert_base_chinese_ner_0329"

app = FastAPI()
templates = Jinja2Templates(directory="./app/templates")


class Input(BaseModel):
    id: int
    sentence: str


class Result(BaseModel):
    entity_group: str
    score: float
    word: str
    start: int
    end: int


class ResultOutput(BaseModel):
    id: int
    content: List[Result]


class Output(BaseModel):
    result: List[ResultOutput]


@app.post("/api/ner", response_model=Output)
def result(input: List[Input]):

    result = []
    for data in input:
        temp = {"id": data.id}
        content = data.sentence
        document = content.replace(" ", "[MASK]")
        document = pred_result(TOKEN, MODEL, document)
        temp["content"] = document
        result.append(temp)

    print(result)
    return {"result": result}


@app.get("/")
def read_root(request: Request, content="", result=""):

    if content != "":
        document = content.replace(" ", "[MASK]")
        document = content.replace("　", "[MASK]")
        result = get_result(TOKEN, MODEL, document)

    response = {
        "request": request,
        "content": content,
        "result": result,
    }
    return templates.TemplateResponse("home.html", response)
