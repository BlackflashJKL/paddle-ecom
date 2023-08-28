# 以事件为中心的观点挖掘

- 基于百度深度学习框架PaddlePaddle
- 使用BERT作为Backbone

## 任务描述

- 由两个子任务构成
  1. 观点抽取：在给定**观点**的条件下，从**文章**中抽取出**表达观点的句子**
  2. 目标提取：从**表达观点的句子**中抽取出**观点目标**（通常是短语）
 
- 输入输出：
  - 输入：观点，文章
  - 输出：表达观点的句子，观点目标
  
- 一个简单的例子：

![image](https://github.com/BlackflashJKL/paddle-ecom/assets/54808417/8155e6c6-b3d0-41e4-8969-d0613bc91599)

## 环境配置

### 方法1：Docker（推荐）

```python
docker pull docker.io/blackflash799/paddle-ecom:v1 # 拉取镜像
docker run -it docker.io/blackflash799/paddle-ecom:v1 /bin/bash # 进入容器
conda activate ecom # 激活环境
git clone # 克隆仓库
```

### 方法2：Anaconda

General
- Python (verified on 3.8)
- CUDA (verified on 11.1)

Python Packages
- see requirements.txt

```python
conda create -n ecom python=3.8 # 创建环境
conda activate ecom # 激活环境
pip install -r requirements.txt # 安装相关依赖
```

## Quick Start
### Data Format
**Additional Statement：** We organize [an evaluation](http://e-com.ac.cn/ccl2022.html/) in CCL2022.

Data folder contains two folders: ECOB-EN and ECOB-ZH.

Before training models, you should first download [data](https://47.94.193.253:25898/down/eac0R1Remzo1.zip) and unzip them as follows. 
```
data
├── ECOB-ZH  # Chinese dataset.
├── ── train.doc.json
├── ── train.ann.json
├── ── dev.doc.json
├── ── dev.ann.json
├── ── test.doc.json
├──   ECOB-EN  # English dataset.
├── ── train.doc.json
├── ── train.ann.json
├── ── dev.doc.json
├── ── dev.ann.json
└── ── test.doc.json
```

The data format is as follows:

In train/dev/test.doc.json, each JSON instance represents a document.
```
{
    "Descriptor": {
        "event_id": (int) event_id,
        "text": "Event descriptor."
    },
    "Doc": {
        "doc_id": (int) doc_id,
        "title": "Title of document.",
        "content": [
            {
                "sent_idx": 0,
                "sent_text": "Raw text of the first sentence."
            },
            {
                "sent_idx": 1,
                "sent_text": "Raw text of the second sentence."
            },
            ...
            {
                "sent_idx": n-1,
                "sent_text": "Raw text of the (n-1)th sentence."
            }
        ]
    }
}
```

In train/dev/test.ann.json, each JSON instance represents an opinion extracted from documents.
```
[
	{
            "event_id": (int) event_id,
            "doc_id": (int) doc_id,
            "start_sent_idx": (int) "Sent idx of first sentence of the opinion.",
            "end_sent_idx": (int) "Sent idx of last sentence of the opinion.",
            "argument": (str) "Event argument (opinion target) of the opinion."
  	}
]
```

### Model Training

#### Step 1: Event-Oriented Opinion Extraction

##### Seq

```python
python eoe_model/seq/main.py \
       --lr 5e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --bert bert-base-chinese \
       --num_epochs 10 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/
```
- ```--bert``` refers to pretrained model path. If you want to train model on English dataset, input 'bert-base-cased'.
- ```--data_dir``` refers to data path. If you want to train model on English dataset, input 'data/ECOB-EN/'.
- ```--model_dir``` refers to the path where the model saved.
- ```--result_dir``` refers to the path where the result saved.

##### PairCls
```python
python eoe_model/paircls/main.py \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
```

####  Step 2: Opinion Target Extraction
##### MRC
```python
python ote_model/mrc/main.py \
      --model_name_or_path luhua/chinese_pretrain_mrc_roberta_wwm_ext_large \
      --do_train \
      --do_eval \
      --do_lower_case \
      --learning_rate 3e-5 \
      --num_train_epochs 5 \
      --per_gpu_eval_batch_size=4 \
      --per_gpu_train_batch_size=6 \
      --evaluate_during_training \
      --output_dir model_files/chinese_model/mrc/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/
```
- ```--model_name_or_path``` refers to pretrained model path. If you want to train model on English dataset, input 'bert-large-uncased-whole-word-masking-finetuned-squad'.

##### SpanR
```python
python ote_model/enum/main.py \
      --lr 1e-5 \
      --batch_size 16 \
      --retrain 1 \
      --bert bert-base-chinese \
      --opinion_level segment \
      --ratio 2 \
      --num_epochs 5 \
      --model_folder model_files/chinese_model/ \
      --data_dir data/ECOB-ZH/ \
      --result_dir result/chinese_result/
```
- ```--ratio``` refers to negative sampling ratio. If you want to train model on English dataset, input '5'.

### Model Evaluation

```python
python eval/eval.py \
      --gold_file data/ECOB-ZH/test.ann.json \
      --pred_file result/chinese_result/pred.ann.json
```
## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
