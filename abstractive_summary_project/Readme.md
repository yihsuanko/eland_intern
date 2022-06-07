# Abstractive Summary Project

利用mT5模型分別對台灣新聞資料和紐約時報資料進行訓練。

1. 台灣新聞資料: 訓練標題用
2. 紐約時報資料: 訓練摘要用

- [Useage](#Useage)
- [訓練模型](#訓練模型)
    - [檔案介紹](#檔案介紹)
    - [參數介紹](#參數介紹)
    - [程式建置](#程式建置)
    - [注意事項](#注意事項)
- [api製作和使用](#api製作和使用)
    - [docker](#docker)
- [Tips](#tips)
- [相關資料](#相關資料)

## Useage

使用api

```python
    uvicorn app.main:app --reload
```

- title
    - URL: `/`
    - method: `get/post`
        - Headers:
            - `Content-Type`: `text/html; charset=utf-8`
        - HTML `<form>` 使用post，使網址不會因為input改變
        - 可調整變數
            - sample: `True`:抽樣 `False`:選機率最大
            - num_return_sequences: 產生的句子數量（因為num_beams=10，如果想要產出超過10個句子，需要調整num_beams）

- summary
    - URL: `/summary`
    - method: `get/post`
        - Headers:
            - `Content-Type`: `text/html; charset=utf-8`
        - HTML `<form>` 使用post，使網址不會因為input改變
        - 可調整變數
            - sample: `True`:抽樣 `False`:選機率最大
            - num_return_sequences: 產生的句子數量（因為num_beams=10，如果想要產出超過10個句子，需要調整num_beams）

- post
    - URL: `/api/sum`
    - method: `post`
    - Example Request Body:
        ```json
        [
            {
                "id": 0,
                "content": "string",
                "do_sample": false,
                "num_return_sequences": 1
            }
        ]
        ```


