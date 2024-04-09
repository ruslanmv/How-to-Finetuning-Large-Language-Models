# How to Fine-tune Large Language Models

Hello everyone, today we are going to get started with Fine-tuning Large Language Models (LLMs) locally and in the cloud.

## Introduction
Currently, there are two common methods to customize our LLMs.

### Prompting
- Generic, side project prototypes
  - Pros: No data required to get started, smaller upfront cost, no technical knowledge needed, connect data through retrieval (RAG)
  - Cons: Much less data fits, forgets data, hallucinations, RAG misses or gets incorrect data

### Fine-tuning
- Domain-specific, enterprise, production usage, privacy
  - Pros: Nearly unlimited data fits, learn new information, correct incorrect information, less cost if smaller model, use RAG too, reduce hallucinations
  - Cons: Need high quality data, upfront compute cost, needs technical knowledge for the data

Fine-tuning usually refers to training further. It can be done with self-supervised unlabeled data or labeled data that you curated. It requires much less data compared to fine-tuning for generative tasks, which is not well defined. Fine-tuning updates the entire model, not just a part of it.


## Environment Setup

## Local Setup
First, we are going to install our environment with Python 3.10.11. After you have installed Python in your working directory, you can create your virtual environment using the following command:

```
python -m venv .venv

```
You'll notice a new directory in your current working directory with the same name as your virtual environment. Then, activate the virtual environment:
```
.venv\Scripts\activate.bat

```

It is convenient to have the latest pip installed:

```
python -m pip install --upgrade pip

```
Next, we install Jupyter Notebook, as you can also use it:


```
pip install ipykernel notebook
```


### Step 3: Setup libraries
Once we have our running environment, we install our kernel:



```
python -m ipykernel install --user --name LLM --display-name "Python (LLM)"
```

Then, we will require PyTorch, HuggingFace, and Llama libraries:


```
pip install datasets==2.14.6 transformers==4.31.0 torch torchvision  lamini==2.0.1 ipywidgets python-dotenv sacrebleu sqlitedict omegaconf pycountry rouge_score peft pytablewriter
```

If we are on Windows, we can use the following command:
```
pip install transformers[torch] 
```
If we are on Linux, we can use the following command:
```
pip install accelerate -U
```


## Data Preparation



## First-time Fine-tuning
To start fine-tuning, we need to identify the tasks by bottom-up engineering of a large LLM. Find tasks that the LLM is doing okay at. Pick one task and gather around 1000 inputs and outputs for that task. Then, fine-tune a small LLM on this data.


# What is Instruction Tuning?
Instruction tuning teaches the model to behave more like a chatbot, providing a better user interface for model generation. For example, it turned GPT-3 into ChatGPT, increasing AI adoption from thousands of researchers to millions of people. You can use instruction-following datasets, such as FAWS, customer support conversations, slack messages, etc. If you don't have QA data, you can convert it to QA by using a prompt template or another LLM. The standard cycle of fine-tuning consists of Data Preparation, Training, and Evaluation.


[For more information ](./1_Data_Preparation/README.md)


## Training

Training an LLM is similar to training a neural network. The process involves:
- Adding the training data
- Calculating loss
- Backpropagating through the model
- Updating weights
- Hyperparameters (Learning Rate, Learning Rate Scheduler)

## Training Infrastructure
For real training, we need to consider the amount of parameters required to train. Here is a table showing the AWS Instance, GPU, GPU Memory, Max Inference size (#params), and Max training size (#tokens):

AWS Instance | GPU |GPU Memory|Max Inference size (#params)| Max training size (#tokens)

| AWS Instance   | GPU     | GPU Memory | Max Inference size (#params) | Max training size (#tokens) |
|----------------|---------|------------|-----------------------------|-----------------------------|
| p3.2xlarge     | 1 V100  | 16GB       | 7B                          | 1B                          |
| p3.8xlarge     | 4 V100  | 64GB       | 7B                          | 1B                          |
| p3.16xlarge    | 8 V100  | 128GB      | 7B                          | 1B                          |
| p3dn.24xlarge  | 8 V100  | 256GB      | 14B                         | 2B                          |
| p4d.24xlarge   | 8 A100  | 320GB      | 18B                         | 2.5B                        |
| p4de.24xlarge  | 8 A100  | 640GB      | 32B                         | 5B                          |

The instruction tuned, teaches the model to behave more like a chatbot, better user interface for model generation. For example, turned GPT-3 into ChatGPT, increase AI adoption, from thousandss of reseachers to millions of people.



[For more information ](./2_Training/README.md)

## Evaluation
There are several ways to evaluate the results of training, but it requires having good test data that is of high quality, accurate, and generalized, not seen in the training data. Currently, there is an Elo comparison. LLM benchmarks like ARC (a set of grade-school questions) and HellaSwag - MMLU (multitask metrics covering elementary math, US history, computer science, law, and more) can be used. TrufulQA is another benchmark.

## Error Analysis
Error analysis involves understanding the behavior of the base model before fine-tuning. Categorize errors and iterate on data to fix these problems in the data space.

[For more information ](./3_Evaluation/README.md)




## PEFT  Parameter-Efficient Finetuning
PEFT stands for Parameter-Efficient Fine-tuning. It refers to the process of fine-tuning LLMs with fewer trainable parameters, resulting in reduced GPU memory usage and slightly lower accuracy compared to fine-tuning all parameters. PEFT involves training new weights in some layers and freezing main weights. It uses low-rank decomposition matrices of the original weights to make changes. During inference, it merges the new weights with the main weights.

## LORA  Low Rank Adaptiaion of LLMs
LORA is another approach for adapting LLMs to new, different tasks. It also involves training new weights in some layers and freezing main weights. LORA uses LoRa for adaptation to new tasks.

## Why Fine-tune All the Parameters?
- LORA (Low Rank Adaptation of LLMs)
  - Fewer trainable parameters for GPT3 (1000x less)
  - Less GPU memory usage
  - Slightly lower accuracy compared to fine-tuning
  - Same inference latency
  - Train new weights in some layers and freeze main weights
  - New weights are rank decomposition matrices of original weights
  - Merge with main weights during inference

## Conclusion

In this blog post, we have discussed how to fine-tune large language models. We have covered the different methods of customization, environment setup, instruction tuning, training, evaluation, error analysis, training infrastructure, PEFT, LORA, and the importance of fine-tuning all parameters. Fine-tuning large language models can greatly benefit enterprises in improving their language models and achieving better results in various tasks.