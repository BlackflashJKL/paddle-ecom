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

![image](https://github.com/BlackflashJKL/paddle-ecom/assets/54808417/6bf4a303-3787-4c9e-983c-1bcbfd5b4b87)

## 项目和数据集结构

### 项目结构

```
paddle-ecom
├── data              # 数据集
├── eoe_model         # 观点抽取模型代码
│   ├── paircls
│   └── seq
├── ote_model         # 目标提取模型代码
│   ├── enum_paddle
│   └── mrc_paddle
├── scripts           # 训练脚本
│   ├── MRC.sh
│   ├── PairCls.sh
│   ├── Seq.sh
│   └── SpanR.sh
├── inference         # 可视化推理脚本
│   ├── data
│   ├── opinion_inferencer.py
│   └── results
├── eval              # 模型评估脚本
│   ├── end2end.py
│   ├── end2end.sh
│   ├── eoe_eval.py
│   ├── eoe_eval.sh
│   ├── eval.py
│   ├── ote_eval.py
│   ├── ote_eval.sh
│   └── overlapf1.py
├── model_files       # 模型保存
│   ├── chinese_model
│   └── english_model
├──  result           # 推理结果保存
│   └── chinese_result
│   └── english_model
└── requirements.txt

```

### 数据集

```
data
├── ECOB-EN  # 中文数据集
│   ├── dev.ann.json
│   ├── dev.doc.json
│   ├── test.ann.json
│   ├── test.doc.json
│   ├── train.ann.json
│   └── train.doc.json
├── ECOB-ZH  # 英文数据集
│   ├── dev.ann.json
│   ├── dev.doc.json
│   ├── test.ann.json
│   ├── test.doc.json
│   ├── train.ann.json
│   └── train.doc.json
└── README.md
```

数据格式详见[数据集介绍](data/README.md)。

## 环境配置

### 方法1：Docker（推荐）

```python
docker pull blackflash799/paddle-ecom:v1 # 拉取镜像
docker run -it --gpus all blackflash799/paddle-ecom:v1 /bin/bash # 进入容器
conda activate ecom # 激活环境
git clone git@github.com:BlackflashJKL/paddle-ecom.git # 克隆仓库
```

### 方法2：Anaconda

```python
conda create -n ecom python=3.8 # 创建环境
conda activate ecom # 激活环境
pip install -r requirements.txt # 安装相关依赖
git clone git@github.com:BlackflashJKL/paddle-ecom.git # 克隆仓库
```

## 运行

### 模型训练

#### 第一步: 观点抽取

##### PairCls

```python
python eoe_model/paircls/main.py \
       --device gpu:3 \
       --bert bert-base-chinese \
       --lr 5e-6 \
       --batch_size 24 \
       --retrain 1 \
       --num_epochs 10 \
       --model_folder model_files/chinese_model/ \
       --data_dir data/ECOB-ZH/ \
       --result_dir result/chinese_result/
```

- ```--bert``` refers to pretrained model path. If you want to train model on English dataset, input 'bert-base-cased'.
- ```--data_dir``` refers to data path. If you want to train model on English dataset, input 'data/ECOB-EN/'.
- ```--model_dir``` refers to the path where the model saved.
- ```--result_dir``` refers to the path where the result saved.

##### Seq

```python
python eoe_model/seq/main.py \
       --device gpu:2 \
       --lr 1e-4 \
       --backbone_lr 1e-6 \
       --batch_size 1 \
       --bert bert-base-chinese \
       --num_epochs 15 \
       --data_dir data/ECOB-ZH/ \
       --model_dir model_files/chinese_model/ \
       --result_dir result/chinese_result/
```

####  第二步：目标提取

##### SpanR

```python
python ote_model/enum_paddle/main.py \
      --device gpu:0 \
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

##### MRC

```python
python ote_model/mrc_paddle/main.py \
      --device gpu:1 \
      --predict_file test \
      --model_name_or_path hfl/roberta-wwm-ext-large \
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

### 模型评估

#### 观点抽取评估

```python
python eval/eoe_eval.py \
    --gold_file data/ECOB-ZH/test.ann.json \
    --pred_file result/chinese_result/seq.pred.json
```

#### 目标提取评估

```python
python eval/ote_eval.py \
    --gold_file data/ECOB-ZH/test.ann.json \
    --pred_file result/chinese_result/mrc.ann.json
```

### 模型推理

```python
python inference/opinion_inferencer.py
```

## License
The code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.
