# eland Internship 學習筆記

本專案紀錄在eland研發二部(RD2)實習的專案內容和心得。這次的實習我參與了兩個與NLP有關的專案分別是「NER專案——政府單位標籤製作」和「自動摘要」。程式碼和使用方式都在資料夾內。

This project records the content and experience of the internship in the RD2 of eland. In this internship, I participated in two NLP-related projects, namely "NER Project - Government Unit Label Production" and "Automatic Summary". The code and usage are in the folder.

- [NLP簡介](#NLP簡介)
    - [Seq2Seq](#Seq2Seq)
- [NER_Project](#NER_Project)
- [Abstractive_summary](#Abstractive_summary)
    - [T5](#T5)
    - [mT5](#mT5)
    - [生成方式演算法介紹](#生成方式演算法介紹)
    - [GPT2](#GPT2)
- [參考資料](#參考資料)

## NLP簡介
自然語言處理（Natural language processing）是一種透過複雜的數學模型及演算法來讓機器去認知、理解並運用我們的語言的技術。

早期的 NLP 技術主要基於統計的概念去訓練模型，讓機器閱讀大量的資料，計算單字、句子出現的機率，然而此種方式無法使系統很好地辨識複雜的文法且產生的內容不流暢。

深度學習的出現，改變了過往訓練 NLP 的運作模式，而目前最廣為研究人員使用的演算法模型即是 BERT， BERT 的全名為轉譯器的雙向編碼表述（Bidirectional Encoder Representations from Transformers）。

BERT 事先透過預先訓練演算法，雙向地去查看前後字詞，進而推斷出完整的上下文，如此的做法不同於以往的模型，能夠更全面的連結上下文，有效幫助系統在文本上的理解與生成。

### Seq2Seq
Seq2Seq是由一個 Encoder 和一個 Decoder 組成。Encoder 獲取輸入序列並將其映射到更高維空間（n 維向量）。該抽象向量被輸入Decoder，Decoder將其轉換為輸出序列。
我們可以把Encoder 和 Decoder想像成只能說兩種語言的人工翻譯。他們的第一語言是他們的母語（例如中文和英文），第二語言是他們共同的想像語言。為了將中文翻譯成英文，Encoder將中文句子轉換成想像語言，再由Decoder讀取該想像語言，翻譯成英文。

由於Encoder 和 Decoder共同會的是想像語言，且在一開始都不是很流利，因此我們需要再翻譯前讓他們做足夠多的學習，也就是預訓練。

此外，與其只把 Encoder 處理完句子產生的最後「一個」向量交給 Decoder 並要求其從中萃取整句資訊，不如將 Encoder 在處理每個詞彙後所生成的「所有」輸出向量都交給 Decoder，讓Decoder 自己決定在生成新序列的時候要把「注意」放在 Encoder 的哪些輸出向量上面。

## NER_Project

實習的第一個專案與專有名詞辨識，或命名實體辨識（Named Entity Recognition, NER）的工作有關，也就是透過機器學習來辨識文本裡面的詞，像是人名、地名、組織等等......

![image](./image/ner1.png)

在本專案，我們會將政府單位從組織中獨立出來，製作政府單位的標籤。

## Abstractive_summary

在自動摘要的專案我們使用了兩種模型（mT5和GPT2）。

mT5是多國語言版本的T5，特性與T5相似，因此我們會先介紹T5。

### T5
T5是Text-To-Text Transfer Transformer的簡稱，資料先進行預訓練，在使用預訓練的模型參數對真正目標領域進行微調(fine-tuning)。T5適用在許多NLP相關的工作，像是翻譯、分類、回歸（例如，預測兩個句子的相似程度，相似度分數在 1 到 5 之間），其他seq2seq任務，如摘要、生成文章。

T5 在預訓練過程中使用C4 (Colossal Clean Crawled Corpus)的資料，C4是透過將網頁文章爬取下來後，刪除重複數據、不完整的句子使資料庫足夠乾淨，預訓練時把C4資料集以corrupted spans方式進行

與GPT2不同的是，T5包含Encoder 和 Decoder，而GPT2只有Decoder。

T5能做的事
- 翻譯
- 問答
- 分類
- 摘要
- 回歸

怎麼做到摘要

decoder被訓練來預測給定前一個單詞的序列中的下一個單詞。

以下是解碼測試序列的步驟：

1. 編碼整個輸入序列並使用編碼器的內部狀態初始化解碼器
2. 將< start > 標記作為輸入傳遞給解碼器
3. 使用內部狀態運行解碼器一個時間步長
4. 輸出將是下一個單詞的概率。將選擇概率最大的單詞
5. 在下一個時間步中將採樣的單詞作為輸入傳遞給解碼器，並使用當前時間步更新內部狀態
6. 重複步驟 3 - 5，直到我們生成 < end > 標記或達到目標序列的最大長度

### mT5

mT5與T5不同的是需要fine-tuned才能使用，在single-task fine-tuning不需要使用前綴，但在multi-task fine-tuning需要使用前綴。

mT5 在mC4語料庫上進行了預訓練，涵蓋 101 種語言

### 生成方式演算法介紹

生成的摘要會因為演算法的不同而有不同的結果，以下介紹pipeline可以使用的4種演算法。

- Greedy search
    - `do_sample = False`
    - `num_beams = 1`
    - 走機率最大的
    - 最簡單的演算法，但生產出來的內容，受限於訓練資料。
- Random sampling
    - `do_sample = True`
    - `num_beams = 1`
    - 依照字的機率，隨機抽一個
    - 可以搭配`temperature`使用，`temperature` 會增加機率大的字的機率，減少機率小的字的機率. 
    - `temperature = 1` -> Random sampling
    - `0 < temperature < 1` -> 字的機率會做調整，越小效果越大
    - `temperature -> 0` -> Greedy search
    - 可以搭配`top_k`、`top_p`使用
    - `top_p` 可以解決使用`top_k`每次都會有一定數量的候選字
- Beams search
    - `do_sample = False`
    - `num_beams > 1`
    - 會保留前幾名，直到結束，可以用來產生多個結果。
    - 適用在每次輸出長度都差不多的情況
    - 嚴重受到重複生成的影響
    - 隨機性不足，與人類日常不同

- Beams sampling
    - `do_sample = True`
    - `num_beams > 1`
    - 將兩種方法結合使用
    - mt5使用的過程中，如果有使用topk,temperature容易會有其他語言出現

### GPT2

GPT-2 的前身是 GPT，其全名為 Generative Pre-Training。在 GPT-2 的論文裡頭，作者們首先從網路上爬了將近 40 GB，名為 WebText（開源版） 的文本數據，並用此龐大文本訓練了數個以 Transformer 架構為基底的語言模型（language model），讓這些模型在讀進一段文本之後，能夠預測下一個字（word）。

值得一提的是，OpenAI 提出的 GPT 跟 Google 的語言代表模型 BERT 都信奉著兩階段遷移學習：利用大量文本訓練出一個通用、具有高度自然語言理解能力的 NLP 模型。有了一個這樣的通用模型之後，之後就能透過簡單地微調同一個模型來解決各式各樣的 NLP 任務，而無需每次都為不同任務設計特定的神經網路架構，省時省力有效率。

兩者的差別則在於進行無監督式訓練時選用的訓練目標以及使用的模型有所不同：

- GPT 選擇 Transformer 裡的 Decoder，訓練目標為一般的語言模型，預測下個字
- BERT 選擇 Transformer 裡的 Encoder，訓練目標則為克漏字填空以及下句預測

GPT-2 也被運用在閱讀理解、翻譯、摘要以及問答任務等多種NLP任務。像是 This Waifu Does Not Exist 在使用 GAN 生成動漫頭像的同時也利用 GPT-2 隨機生成一段動漫劇情；而 TabNine 則是一個加拿大團隊利用 GPT-2 做智慧 auto-complete 的開發工具，志在讓工程師們減少不必要的打字，甚至推薦更好的寫法。

## 參考資料
1. [T5參考影片 -> Colin Raffel](https://www.youtube.com/watch?v=eKqWC577WlI&list=UUEqgmyWChwvt6MFGGlmUQCQ&index=5)
2. [淺談神經機器翻譯 & 用 Transformer 與 TensorFlow 2](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html?fbclid=IwAR2eHxhPxyg96A3mbtveRHd5zFKscSLA-u8jdoDueUC9Dl1g3Vrv-61Y84g)
3. [Decoding Strategies that You Need to Know for Response Generation](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)

4. [直觀理解GPT2](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html)
