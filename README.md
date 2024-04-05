# How to Finetunning Large Language Models

Hello everyone, today we are going getting started with Finetunning Large Language Models in the Cloud.


## Introduction

Currenttly there are two common different methods to to custom our LLMS.
## Prompting
Generic , side projects prototypes

Pros
- No data to get started
- Smaller upfront cost
- No techniical knowledge neeeded
- Connect data through retrieval (RAG)

Cons
- Much less data fits
- Forgets data
- Hallucinations
- RAG misses or gets incorrect data

## Finetunning


Domain specific, enterprise, production usage, privacy.

Pros
- Nearly unlimited data fits
- Learn new information
- Correct incorrect information
- Less cost if smaller model
- Use RAG too
- Reduce Hallucitnations

Cos
- Need high quality data
- Upfront compute cost
- Needs technical knowledge for the data.


Fine tunning usually refers to traineing furter, can also be self-supervissed unlabeled data, can labeled data you curated, much less data needed.
Fine tunning for generative task is not weell defined. Updates entire model, non gust part of it.


## Environment Setup



## Local Setup
First we are going to install our enviroment with python 3.10.11 here , after you installed in your working directory you can create your enviroment

```
python -m venv .venv

```
You’ll notice a new directory in your current working directory with the same name as your virtual environment, then activate the virtual environment.

```
.venv\Scripts\activate.bat

```

usually is convinent having the latest pip

```
python -m pip install --upgrade pip

```
then we install our notebook, because also you can use Jupyter Notebook

```
pip install ipykernel notebook
```


## Step 3 . Setup libraries
Once we have our running environment we install our kernel


```
python -m ipykernel install --user --name LLM --display-name "Python (LLM)"
```

then we will require Pytorch, HuggingFace and Llama libraries.


```
pip install datasets==2.14.6 transformers==4.31.0 torch torchvision  lamini==2.0.1 ipywidgets python-dotenv
```

If we are in windows we can use
```
pip install transformers[torch] 
```
or in linux
```
pip install accelerate -U
```

## First time fine tunning

We identify the tasks by bompt engineering of a large LLM.

Find tasks that you see an LLM doing OK at.

Pick one task.

Get near 1000 inputs and out puts for the task
Finetune a small LLM on this data.


# What is Instruction tunning

The instruction tuned, teaches the model to behave more like a chatbot, better user interface for model generation. For example, turned GPT-3 into ChatGPT, increase AI adoption, from thousandss of reseachers to millions of people.

You can use Instruction-following datasets, by using FAWS, customer support conversations, slack messages etc.

If you dont have QA data you can convert to QA by using a prompt template or using another llm.

The standard cycle of Finetineing is Data Preparation, Traniing and Evaluation


# Training 

What is training model in LLM

The process to train LLMS is the same as a neural network. 
- First you need to add the training data
- Calculate loss
- Backprop trhough model
- Update weights
  
  Hyperparaters
- Learning Rate
- Learner rate schedular
