from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Specify the local model path
model_name_or_path = "/Users/oktayosman/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/221e3535e1ac4840bdf061a12b634139c84e144c"
fine_tuned_model_path = 'models/fine-tuned-model'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map=None,
)
model.resize_token_embeddings(len(tokenizer))

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset('json', data_files={
    'train': 'data/train.jsonl',
    'validation': 'data/validation.jsonl'
})

# Define the function to format each example
def format_example(example):
    # Format the prompt
    prompt = f"""### Instruction:
{example['instruction']}

### Job Description:
{example['job_description']}

### CV:
{example['cv']}

### Response:
"""
    # Format the response
    if isinstance(example['response'], dict):
        formatted_response = f"""Match Percentage:
{example['response'].get('match_percentage', '')}%

Good Impressions:
{example['response'].get('good_impression', '')}

Warnings:
{example['response'].get('warning', '')}

Bad Impressions:
{example['response'].get('bad_impression', '')}"""
    else:
        formatted_response = example['response']

    return prompt, formatted_response

# Tokenize the dataset
def tokenize_function(examples):
    input_ids_list = []
    labels_list = []
    for i in range(len(examples['instruction'])):
        example = {
            'instruction': examples['instruction'][i],
            'job_description': examples['job_description'][i],
            'cv': examples['cv'][i],
            'response': examples['response'][i]
        }
        prompt, response = format_example(example)

        # Tokenize prompt and response separately
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None,
        )

        response_tokens = tokenizer(
            response,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None,
        )

        # Concatenate prompt and response
        input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']

        # Create labels: -100 for prompt tokens, actual ids for response tokens
        labels = [-100] * len(prompt_tokens['input_ids']) + response_tokens['input_ids']

        # Truncate to max_length
        if len(input_ids) > 1024:
            input_ids = input_ids[:1024]
            labels = labels[:1024]

        # Pad input_ids and labels to max_length
        padding_length = 1024 - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        labels += [-100] * padding_length

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {'input_ids': input_ids_list, 'labels': labels_list}

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,  # Increased epochs due to small dataset
    per_device_train_batch_size=2,  # Increase if GPU memory allows
    per_device_eval_batch_size=2,
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    save_strategy='epoch',        # Save model at the end of each epoch
    logging_steps=10,
    learning_rate=5e-5,  # Increased learning rate for small dataset
    save_total_limit=2,
    weight_decay=0.01,
    warmup_steps=50,
    load_best_model_at_end=True,          # Added this line
    metric_for_best_model='eval_loss',    # Optional: specify the metric to monitor
    greater_is_better=False,              # Since lower eval_loss is better
)

# Use the DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Early stopping defined here
)

# Train the model
trainer.train()

# After training, save the fine-tuned model and tokenizer
trainer.save_model(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)