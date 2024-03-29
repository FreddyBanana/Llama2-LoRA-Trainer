# Llama2-LoRA-Trainer

## 简介 / Introduction

本仓库利用[peft](https://github.com/huggingface/peft)库与transformers.Trainer，实现对Llama2的简单4-bit/8-bit LoRA微调。

This repo uses [peft](https://github.com/huggingface/peft) and transformers.Trainer to achieve simple 4-bit/8-bit LoRA fine-tuning for Llama2. 



## 安装依赖 / Installing the dependencies

```
$ pip install -r requirement.txt
```

requirement.txt：

```
datasets==2.13.1
protobuf
transformers==4.30.2
cpm_kernels
torch>=2.0
mdtex2html
sentencepiece
accelerate
git+https://github.com/huggingface/peft.git
bitsandbytes
loralib
scipy
```



## 参数/config

文件config.py参数如下：

- **MICRO_BATCH_SIZE**，每块GPU的batch size大小。
- **BATCH_SIZE**，真正的batch size，当每个batch的处理样本数达到BATCH_SIZE时，进行梯度更新。
- **EPOCHS**，总训练代数。
- **WARMUP_STEPS**，预热步数。
- **LEARNING_RATE**，学习率。
- **CONTEXT_LEN**，context字段截断长度（对应json文件的context）。
- **TARGET_LEN**，target字段截断长度（对应json文件的target）。
- **TEXT_LEN**，text字段截断长度（对应txt文件的文本）。
- **LORA_R**，LoRA低秩的秩数。
- **LORA_ALPHA**，LoRA的alpha。
- **LORA_DROPOUT**，LoRA层的Dropout率。
- **MODEL_NAME**，模型名称（huggingface仓库地址）。
- **LOGGING_STEPS**，日志步数，即训练的时候输出loss的间隔步数。
- **OUTPUT_DIR**，输出LoRA权重的存放文件夹位置。
- **DATA_PATH**，数据集文件位置。
- **DATA_TYPE**，数据集文件类型，可选json或txt。
- **SAVE_STEPS**，保存LoRA权重的间隔步数。
- **SAVE_TOTAL_LIMIT**，保存LoRA权重文件的总数（不包括最终权重）。
- **PROMPT**，推理时的prompt。
- **TEMPERATURE**，推理时的温度，调整模型的创造力。
- **LORA_CHECKPOINT_DIR**，待推理LoRA权重的文件夹位置。
- **BIT_4**，使用4bit量化+LoRA微调。
- **BIT_8**，使用8bit量化+LoRA微调。



The parameters in config.py are as follows:

- **MICRO_BATCH_SIZE**，batch size on each device。
- **BATCH_SIZE**，when the number of processed samples in each split batch reaches BATCH_SIZE, update the gradient.
- **EPOCHS**，training epochs。
- **WARMUP_STEPS**，warmup steps。
- **LEARNING_RATE**，learning rate of fine-tuning。
- **CONTEXT_LEN**，truncation length of context (in json)。
- **TARGET_LEN**，truncation length of target (in json)。
- **TEXT_LEN**，truncation length of text (in txt)。
- **LORA_R**，LoRA low rank。
- **LORA_ALPHA**，LoRA Alpha。
- **LORA_DROPOUT**，LoRA dropout。
- **MODEL_NAME**，model name (huggingface repo address)。
- **LOGGING_STEPS**，the number of interval steps for outputting loss during training。
- **OUTPUT_DIR**，the storage folder location for LoRA weights。
- **DATA_PATH**，the location of your dataset file。
- **DATA_TYPE**，the type of your dataset file, including json and txt。
- **SAVE_STEPS**，the number of interval steps to save LoRA weights。
- **SAVE_TOTAL_LIMIT**，the total number of LoRA weight files saved (excluding the final one)。
- **PROMPT**，your prompt when inference。
- **TEMPERATURE**，the temperature when inference, adjusting the creativity of LLM。
- **LORA_CHECKPOINT_DIR**，folder location for LoRA weights to be inferred。
- **BIT_4**，use 4-bit。
- **BIT_8**，use 8-bit。



## 数据集文件/Dataset files

### 1）json

json文件格式如下：

The JSON file format is as follows：

```
{"context":question1, "target":answer1}{"context":question2, "target":answer2}...
```



### 2）txt

txt文件格式如下：

The txt file format is as follows：

```
sentence1
sentence2
sentence3
...
```



## 使用方法 / Usage

### 1）训练/train

```
$ sh train.sh
```

train.sh：

```shell
python main.py \
	--MICRO_BATCH_SIZE 8 \
	--BATCH_SIZE 16 \
	--EPOCHS 10 \
	--LEARNING_RATE 5e-6 \
	--CONTEXT_LEN 64 \
	--TARGET_LEN 192 \
	--LORA_R 16 \
	--LORA_DROPOUT 0.2 \
	--MODEL_NAME TheBloke/Llama-2-7B-fp16 \
	--OUTPUT_DIR ./output_model \
	--DATA_PATH ./new_train.json \
	--DATA_TYPE json \
	--SAVE_STEPS 1000 \
	--BIT_4
```



### 2）推理/inference

```
$ sh inference.sh
```

inference.sh：

```shell
python inference.py \
	--CONTEXT_LEN 256 \
	--MODEL_NAME TheBloke/Llama-2-7B-fp16 \
	--LORA_CHECKPOINT_DIR ./output_model/model_final \
	--BIT_4 \
	--PROMPT "put your prompt here"
```



## 参考 / Reference

[Fine_Tuning_LLama | Kaggle](https://www.kaggle.com/code/gunman02/fine-tuning-llama?scriptVersionId=128204744)

[FreddyBanana/ChatGLM2-LoRA-Trainer: Simple 4-bit/8-bit LoRA fine-tuning for ChatGLM2 with peft and transformers.Trainer. (github.com)](https://github.com/FreddyBanana/ChatGLM2-LoRA-Trainer)

[mymusise/ChatGLM-Tuning: 一种平价的chatgpt实现方案, 基于ChatGLM-6B + LoRA (github.com)](https://github.com/mymusise/ChatGLM-Tuning/tree/master)