# NER_project

實作NER政府單位標籤，利用中研院的Albert(albert-base-chinese, albert-tiny-chinese)模型進行訓練。
可以到 [Hugging Face Hub](https://huggingface.co/models) 找其他模型來使用。

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

1. 使用Hugging Face Hub 的[yihsuan/albert-base-chinese-0407-ner](https://huggingface.co/yihsuan/albert-base-chinese-0407-ner)模型。

2. 執行程式碼，如果Huggingface 有author限制，可以直接[yihsuan/albert-base-chinese-0407-ner](https://huggingface.co/yihsuan/albert-base-chinese-0407-ner)下載模型。

    ```python
        python3 app/ner_predict.py
    ```
    -> 得到專有標籤的list

3. 使用api

    ```python
        uvicorn app.main:app --reload
    ```
    - URL:/
        - method:get
    - URL:/api/ner
        - method:post
        - Example Request Body:
        ```python
            [
                {
                    "id": 0,
                    "sentence": "三峽親友及校園案40例最多"
                },
                {
                    "id": 1,
                    "sentence": "小明想要去台北玩"
                }
            ]
        ```


## 訓練模型

```python
    python3 run_ner.py run_ner_config.json
```

### 檔案介紹

- `data_preprocessing.py` : 整理政府組織檔
- `trainset_prep.py` : 預處理train file檔案
- `run_ner.py`: 主要程式檔
- `run_ner_config.json`: 參數設定檔

由於是使用已有標籤(人名、地名、組織)的資料進行預測，因此在預處理train file的過程只進行政府單位的定義，且定義方式是將原本的組織標籤進行重新判定，如果是政府組織會改為政府組織標籤。

政府組織的列表由政府資料開放平臺取得。

### 參數介紹

參數均設定在`run_ner_config.json`。

重要參數介紹
- `model_name_or_path`: 模型路徑
- `tokenizer_name`: 要使用的文本預處理方式
- `metric_for_best_model`: 模型評估方式
- `return_entity_level_metrics`: 輸出評估矩陣

### 程式建置

[詳細程式建置和模型評估](https://github.com/yihsuanko/gov_ner_project)

### 注意事項

- ner label 包含（Ｏ, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-GORG(新增), I-GORG(新增)）

- 其中Ｏ是沒被定義得標籤，B為標籤的開頭，I為接續B的標籤。

- 如果使用模型內建的`return_entity_level_metrics`，要注意在假設沒有 B 標籤時, 程式碼會把第一個 I 標籤換成 B 標籤，會與另外自己將 B 和 I 的標籤合併時的precision和recall有落差。

## api製作和使用

[詳細此api介紹](https://github.com/yihsuanko/gov_ner_project)

### [docker](https://www.docker.com/)
- 為什麼要使用docker?<br>
因為每台電腦的作業系統與硬體配置有可能不同，我的程式碼可能剛好只跟我電腦上的環境相容，因此在其他的電腦使用時可能會爆掉、無法使用。
- 如何使用docker
    1. 安裝 Docker
    2. 準備打包資料和requirements.txt<br>
    將所有需要的套件寫入requirements.txt
    ```
    # 舉例
    fastapi>=0.68.0,<0.69.0
    pydantic>=1.8.0,<2.0.0
    uvicorn>=0.15.0,<0.16.0
    ```

    資料放置方式以本專案為例
    ```
    .
    ├── app
    │   ├── __init__.py
    │   ├── ner_predict.py
    │   ├── main.py
    │   └── templates
    ├── Dockerfile
    └── requirements.txt

    ```
    3. 撰寫 Dockerfile
    ```docker
    FROM python:3.9

    WORKDIR /code
    
    COPY ./requirements.txt /code/requirements.txt

    RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt # 環境建置

    COPY ./app /code/app # copy 包含主要程式的資料夾
    COPY ./albert_base_chinese_ner_0329 /code/albert_base_chinese_ner_0329 # copy 模型

    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

    ```
    4. Docker build<br>
    `docker build -t name(自己設定的image):tag(自己設定的標籤，也可以不填) .`<br>

    5. Docker run<br>
    `docker run -d --name mycontainer(自己設定的container) -p 80:80 myimage(自己設定的image)`<br>

## Tips

- 訓練注意事項 :
    1. 訓練資料內的空白字元不會被訓練，因此資料預處理時會忽略空白字元的 Token
        - 預測時可以透過`[MASK]`來取代空白字元的 Token
    2. 當調高 Epoch數時, 可以使得 Evaluation/F1 上升, 但 Evaluation/Loss 也會因此上升，需要留意。
    3. Batch size 建議使用3進行訓練, （也可以嘗試4 or 5），但當提升batch size時Evaluation/Loss 也會因此上升。
    4. 使用pipeline預測時，會造成英文字母無法切分成characters的情況，像是berlin 會分成 "be", "##rlin", 目前pipeline尚未提供解方，因此雖然pipeline比較方便使用，但並不適合在同時含有中英文的文章時使用。
    5. 使用模型的 Tokenizer 需要帶上 `is_split_into_words=True`, 且將訓練資料的每一個字都視為一個 Token，來進行預測和訓練。

- colab 使用需要注意的地方
    1. 如果要使用albert-base, bert-base等模型需要多輸入`!pip install -U transformers`這個指令，才能訓練成功。
    2. 參數的部分不需要有一個檔案，直接在Notebook上更改就行。
    3. 登入Hugging Face可以直接將結果和參數git到Hugging Face，且可以使用Hugging Face的API，將模型視覺化。
- NER 準確度存在迷思 :
  - 準確度高可能是被少數詞彙撐高的, 因為這些詞彙的標記數量多, 於是準確度高
  - 因此可以將那些常出現的詞彙先踢除，來看precision和recall會必較準確

## 相關資料

- [ckiplab/ckip-transformers](https://github.com/ckiplab/ckip-transformers)
- [Fine-tuning a model on a token classification task](https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb)
- [Hugging face Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hugging face Tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)
