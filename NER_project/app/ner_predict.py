from re import I
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import json
from typing import List

def predict(example, ner_results):
    ans = ["O"]*len(example)
    for i in ner_results:
        for j in range(int(i["start"]),int(i["end"])):
            if "B" in i["entity"]:
                if j == int(i["start"]):
                    ans[j] = i['entity']
                else:
                    temp = "I-" + i['entity'][2:]
                    ans[j] = temp
            else:
                ans[j] = i['entity']
    
    return ans

def pred_result(token,model, word):
    tokenizer = AutoTokenizer.from_pretrained(token)
    model = AutoModelForTokenClassification.from_pretrained(model)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") #ignore_labels=[""]
    # nlp2 = pipeline("ner", model=model, tokenizer=tokenizer,ignore_labels=[""])

    ner_results = nlp(word)
    for data in ner_results:
        if isinstance(data["word"], list):
            data["word"] = "".join(data["word"])

        temp = data["word"].replace(" ", "")
        temp = temp.replace("[MASK]", " ")
        data["word"] = temp

    # ner_results = nlp2(word)
    # result = predict(word, ner_results)
    
    return ner_results

def get_result(token,model, word):
    tokenizer = AutoTokenizer.from_pretrained(token)
    model = AutoModelForTokenClassification.from_pretrained(model)

    nlp = pipeline("ner", model=model, tokenizer=tokenizer,ignore_labels=[""], aggregation_strategy="simple")

    ner_results = nlp(word)
    for data in ner_results:
        if isinstance(data["word"], list):
            data["word"] = "".join(data["word"])

        temp = data["word"].replace(" ", "")
        temp = temp.replace("[MASK]", " ")
        data["word"] = temp

    return ner_results

if __name__ == '__main__':
    # load model and tokenizer
    
    token = "albert_base_chinese_ner_0329"
    model = "albert_base_chinese_ner_0329"

    word = "台東地檢署 21日指揮警方前往張靜的事務所及黃姓女友所經營的按摩店進行搜索"
    word = word.replace(" ","[MASK]")
    result = pred_result(token, model, word)
    print(result)