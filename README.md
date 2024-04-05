# How to Finetunning Large Language Models

Hello everyone, today we are going getting started with Finetunning Large Language Models in the Cloud.


## Introduction

Currenttly there are two common different methods to to custom our LLMS 

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


## Environment Setup
For this project  we are going to use AWS Cloud , we are going to use SageMaker Notebook.


First we are going to install our enviroment with python 3.10.11 here , after you installed in your working directory you can create your enviroment

```
python -m venv .venv

```


You’ll notice a new directory in your current working directory with the same name as your virtual environment, then activate the virtual environment.

```
.venv\Scripts\activate.bat

```

usually is convinent having the latest pip

python -m pip install --upgrade pip
then we install our notebook, because also you can use Jupyter Notebook

pip install ipykernel notebook
