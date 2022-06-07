# Abstractive Summary Project

利用mT5模型分別對台灣新聞資料和紐約時報資料進行訓練。

1. 台灣新聞資料: 訓練標題用
2. 紐約時報資料: 訓練摘要用

- [Useage](#Useage)
- [訓練模型](#訓練模型)
    - [檔案介紹](#檔案介紹)
    - [參數介紹](#參數介紹)
    - [生成時參數介紹](#生成時參數介紹)
    - [注意事項](#注意事項)

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

## 訓練模型

```python
    python3 run_summarize.py run_sum_config.json
```

### 檔案介紹

- `run_summarize.py`: 主要程式檔
- `run_sum_config.json`: 參數設定檔


### 參數介紹

`train_file`：csv檔，如果沒有指定文章欄位第一個會視為文章、第二個為摘要。

- DataTrainingArguments
    - `text_column`:文章欄位名稱
    - `summary_column`:摘要欄位名稱 (沒有設定，第一個會視為文章、第二個為摘要)
    - `preprocessing_num_workers`: 加快資料預處理的速度
    - `max_source_length`:文章採取長度（可以無限長）
    - `max_target_length`:摘要採取長度
    - `num_beams`:beams 搜尋法 （用在evaluate and predict）
    - `source_prefix`:任務類別（用在t5訓練）

- TrainingArguments
    - `load_best_model_at_end`: 設定為true時，一定要設定`evaluation_strategy`
    - `evaluation_strategy`:可以是steps、no、epoch，如果不是no，do_eval會自動為true
    - `gradient_accumulation_steps`:累積梯度，「變相」擴大`batch size`。
    - `fp16`:mt5 不能使用

### 生成時參數介紹

- Predict時使用(transformers pipeline)
    ```python
    from transformers import pipeline
    import torch
    torch.manual_seed(32)  # 固定random seed
    summarizer = pipeline("summarization",model="./model")
    summarizer(text,args*)

    ```
    - `max_length`: 生產最長字數限制
    - `min_length`: 生產最短字數限制
    - `repetition_penalty`:疑似t5無法使用
    - `no_repeat_ngram_size`:重複字詞限制
    - `num_beams`:beams 搜尋法
    - `early_stopping`: 當所有beams 找到 EOS token.
    - `do_sample`: 隨機抽樣
    - `top_k`: (defaults to 50) 抽樣限制，先透過機率排序，選出機率最大的K個字，再來分機率，最後抽樣
    - `top_p`:(defaults to 1.0)抽樣限制，先透過機率排序，選出累績機率和等於P的字，再來分機率，最後抽樣

### 注意事項
1. 訓練基礎因使用超過3000筆資料，資料太少預測結果會出現 `<extra_id_0>`
2. 使用mt5時，不能使用`fp16`，會造成訓練問題，導致預測結果不良
3. 因為記憶體問題，`batch size`無法調太大，但在一定條件下，`batch size`越大，模型越穩定。此時可以調`gradient_accumulation_steps`，來解決顯卡儲存空間的問題，如果`gradient_accumulation_steps`為8，則`batch size`「變相」擴大8倍。
