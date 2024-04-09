**Data Preparation for Fine-Tuning Large Language Models**

Large language models (LLMs) are trained on massive amounts of text data to understand and generate human-like language. Fine-tuning takes a pre-trained LLM and tailors it for a specific task using a smaller dataset relevant to that task. This section dives into data preparation, a crucial step for successful fine-tuning.

**1. Pre-training vs. Fine-tuning Data**

- **Pre-training Data:** LLMs are pre-trained on vast, general-purpose datasets like Common Crawl ([https://huggingface.co/datasets/crawl_domain](https://huggingface.co/datasets/crawl_domain)) to grasp general language patterns. 

```python
# This code shows loading the Common Crawl dataset, but it's currently unavailable.
#pretrained_dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)

pretrained_dataset = load_dataset("c4", "en", split="train", streaming=True)

n = 5
print("Pretrained dataset:")
top_n = itertools.islice(pretrained_dataset, n)
for i in top_n:
  print(i)
```

- **Fine-tuning Data:** This is your task-specific dataset that the pre-trained LLM will specialize on. It should be smaller but relevant to the desired outcome.

**2. Contrasting Pre-training and Fine-tuning Data**

The provided code snippet showcases a contrast between the pre-trained dataset (likely containing web pages or books) and your custom fine-tuning dataset (potentially question-answer pairs in JSONL format).

```python
filename = "lamini_docs.jsonl"
instruction_dataset_df = pd.read_json(filename, lines=True)
instruction_dataset_df
```

**3. Formatting Your Fine-tuning Data**

There are various ways to format your data for fine-tuning. Here, the code demonstrates extracting questions and answers from a JSONL file and combining them into a single text format.

```python
examples = instruction_dataset_df.to_dict()
text = examples["question"][0] + examples["answer"][0]
text
```

**4. Creating Prompts**

Prompts provide context and guide the LLM towards the desired task. The code showcases creating prompts for question-answering tasks with placeholders for questions and answers.

```python
prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(question=question, answer=answer)
text_with_prompt_template
```

**5. Saving Your Data**

The code demonstrates saving the processed data with questions and answers in JSONL format for compatibility with various tools.

```python
num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})

with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
  writer.write_all(finetuning_dataset_question_answer)
```




