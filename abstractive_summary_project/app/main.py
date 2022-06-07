import os
from random import sample
from fastapi import FastAPI, Form,  Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from typing import Optional, List, Union
from transformers import pipeline, T5Tokenizer
from pydantic import BaseModel, parse_obj_as
from dotenv import load_dotenv
from pathlib import Path
import string
from zhon import hanzi
import re

load_dotenv()
title_model = os.getenv("TITLE_MODEL")
summary_model = os.getenv("SUMMARY_MODEL")

title_model_path = Path("./models", title_model)
summary_model_path = Path("./models", summary_model)

if not title_model_path.exists():
    raise FileNotFoundError(title_model_path.as_posix())
if not summary_model_path.exists():
    raise FileNotFoundError(summary_model_path.as_posix())

title_model = title_model_path.as_posix()
summary_model = summary_model_path.as_posix()

app = FastAPI()
templates = Jinja2Templates(directory="./app/templates")

summarizer_title = pipeline("summarization", model=title_model)
summarizer_summary = pipeline("summarization", model=summary_model)
tokenizer_title = T5Tokenizer.from_pretrained(title_model)
tokenizer_summary = T5Tokenizer.from_pretrained(summary_model)

punctuation_string = string.punctuation
punctuation_string += " "
punctuation_string += hanzi.punctuation


def clean_punc(text):
    for i in punctuation_string:
        text = text.replace(i, '')
    return text


def pred_title_result(article, max_length=50, num_return_sequences=1, do_sample=False, early_stopping=True, no_repeat_ngram_size=3):

    text = article
    text_v2 = clean_punc(article)
    def prefix_allowed_tokens_fn(batch_id, input_ids): return tokenizer_title.encode(
        text) if len(input_ids) > 3 else tokenizer_title.encode(text_v2)

    if do_sample == True:
        result = summarizer_title(article,
                                  max_length=max_length,
                                  num_beams=10,
                                  do_sample=do_sample,
                                  top_p=0.4,
                                  repetition_penalty=1.2,
                                  early_stopping=early_stopping,
                                  prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                  num_return_sequences=num_return_sequences,
                                  no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        result = summarizer_title(article,
                                  max_length=max_length,
                                  num_beams=20,
                                  num_beam_groups=2,
                                  do_sample=do_sample,
                                  early_stopping=early_stopping,
                                  prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                  repetition_penalty=1.2,
                                  num_return_sequences=num_return_sequences,
                                  no_repeat_ngram_size=no_repeat_ngram_size)

    return result


def pred_summary_result(article, max_length=150, num_return_sequences=1, do_sample=False, early_stopping=True, no_repeat_ngram_size=3):

    text = article
    text_v2 = clean_punc(article)
    def prefix_allowed_tokens_fn(batch_id, input_ids): return tokenizer_summary.encode(
        text) if len(input_ids) > 3 else tokenizer_summary.encode(text_v2)

    if do_sample == True:
        result = summarizer_summary(article,
                                    max_length=max_length,
                                    num_beams=10,
                                    do_sample=do_sample,
                                    top_p=0.4,
                                    repetition_penalty=1.2,
                                    early_stopping=early_stopping,
                                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                    num_return_sequences=num_return_sequences,
                                    no_repeat_ngram_size=no_repeat_ngram_size)
    else:
        result = summarizer_summary(article,
                                    max_length=max_length,
                                    num_beams=20,
                                    num_beam_groups=2,
                                    do_sample=do_sample,
                                    early_stopping=early_stopping,
                                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                    repetition_penalty=1.2,
                                    num_return_sequences=num_return_sequences,
                                    no_repeat_ngram_size=no_repeat_ngram_size)

    return result


class Input(BaseModel):
    id: int
    content: str
    do_sample: Optional[bool] = False
    num_return_sequences: Optional[int] = 1


@app.post("/api/sum")
async def result(input: List[Input]):

    result = []
    for data in input:
        temp = {"id": data.id}
        content = data.content
        temp["content"] = content
        num_return_sequences = data.num_return_sequences
        temp["num_return_sequences"] = num_return_sequences
        do_sample = data.do_sample
        temp["do_sample"] = do_sample
        document = pred_title_result(
            content, do_sample=do_sample, num_return_sequences=num_return_sequences)
        temp["result"] = document
        result.append(temp)

    response = {
        "result": result
    }

    return response  # {"result": result}


@app.post("/")
async def gettitle(request: Request, result="", content: str = Form(...), sample: bool = Form(False), num_return_sequences: str = Form(...)):

    do_sample = False
    if sample:
        do_sample = True

    if num_return_sequences == "":
        num_return_sequences = 1
    else:
        num_return_sequences = int(num_return_sequences)

    if content != "":
        result = pred_title_result(
            content, do_sample=do_sample, num_return_sequences=num_return_sequences)

    for sentence in result:
        temp = sentence["summary_text"]
        temp = temp.replace(",", '，')
        temp = temp.replace(":", '：')
        sentence["summary_text"] = temp

    response = {
        "request": request,
        "content": content,
        "sample": sample,
        "result": result,
        "num_return_sequences": num_return_sequences,
    }

    return templates.TemplateResponse("title.html", response)


@app.get("/")
def title(request: Request, content="", result="", sample="", num_return_sequences=""):
    do_sample = False
    if sample:
        do_sample = True

    if num_return_sequences == "":
        num_return_sequences = 1
    else:
        num_return_sequences = int(num_return_sequences)

    if content != "":
        result = pred_title_result(content,
                                   do_sample=do_sample, num_return_sequences=num_return_sequences)

    for sentence in result:
        temp = sentence["summary_text"]
        temp = temp.replace(",", '，')
        temp = temp.replace(":", '：')
        sentence["summary_text"] = temp

    response = {
        "request": request,
        "content": content,
        "sample": sample,
        "result": result,
        "num_return_sequences": num_return_sequences,
    }
    return templates.TemplateResponse("title.html", response)


@app.post("/summary")
async def getsummary(request: Request, result="", content: str = Form(...), sample: bool = Form(False), num_return_sequences: str = Form(...)):

    do_sample = False
    if sample:
        do_sample = True

    if num_return_sequences == "":
        num_return_sequences = 1
    else:
        num_return_sequences = int(num_return_sequences)

    if content != "":
        result = pred_summary_result(content,
                                     do_sample=do_sample, num_return_sequences=num_return_sequences)

    for sentence in result:
        temp = sentence["summary_text"]
        temp = temp.replace(",", '，')
        temp = temp.replace(":", '：')
        sentence["summary_text"] = temp

    response = {
        "request": request,
        "content": content,
        "sample": sample,
        "result": result,
        "num_return_sequences": num_return_sequences,
    }

    return templates.TemplateResponse("summary.html", response)


@app.get("/summary")
def summary(request: Request, content="", result="", sample="", num_return_sequences=""):
    do_sample = False
    if sample:
        do_sample = True

    if num_return_sequences == "":
        num_return_sequences = 1
    else:
        num_return_sequences = int(num_return_sequences)

    if content != "":
        result = pred_summary_result(content,
                                     do_sample=do_sample, num_return_sequences=num_return_sequences)

    for sentence in result:
        temp = sentence["summary_text"]
        temp = temp.replace(",", '，')
        temp = temp.replace(":", '：')
        sentence["summary_text"] = temp

    response = {
        "request": request,
        "content": content,
        "sample": sample,
        "result": result,
        "num_return_sequences": num_return_sequences,
    }
    return templates.TemplateResponse("summary.html", response)
