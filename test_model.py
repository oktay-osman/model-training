from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datasets import load_dataset
import re

# Paths to the base model and fine-tuned model (update paths for Mac)
model_name_or_path = "/Users/oktayosman/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
fine_tuned_model_path = "/Users/oktayosman/Downloads/fine-tuned-model"

# Force device to CPU for compatibility with Mac
device = torch.device("cpu")
print(f"Using device: {device}")

# Load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

# Load the base model
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map=None,  # Avoid auto-mapping to GPU since we're on CPU
    torch_dtype=torch.float32  # Use float32 for compatibility on CPU
)
print("Model loaded.")

# Load the fine-tuned PEFT model
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.to(device)
print("Fine-tuned model loaded.")

# Set model to evaluation mode
model.eval()

# Load the validation dataset
print("Loading validation dataset...")
dataset = load_dataset('json', data_files={'validation': 'validation.jsonl'})
validation_dataset = dataset['validation']
print("Validation dataset loaded.")

# Function to create a simple prompt for evaluating the CV
def create_prompt(cv_text, job_description):
    prompt = f"""
You are an AI tasked with evaluating a CV in relation to a job description. Your goal is to compare the CV to the job description and provide structured feedback on how well the candidate fits the job requirements.

Provide your evaluation in the following format:

Match Percentage: [0% to 100% — How well the candidate's skills and experience match the job description.]

Good Impressions: [List skills, qualifications, and experience that match the job requirements, and how those skills align with the job description.]

Warnings: [Mention if the candidate lacks some experience, skills, or technologies that are listed as required in the job description. For example, insufficient years of experience or missing non-vital skills.]

Bad Impressions: [Highlight any critical gaps or issues, such as the absence of vital skills or experiences mentioned in the job description.]

**ONLY** provide the response in the format above.

Job Description:
{job_description}

CV:
{cv_text}

Response:
"""
    return prompt

# Function to generate and format model output
def generate_output(cv_text, job_description):
    prompt = create_prompt(cv_text, job_description)

    # Tokenize and move inputs to the device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response section after "Response:"
    response_start = generated_text.find("Response:")
    response_text = generated_text[response_start + len("Response:"):].strip() if response_start != -1 else generated_text.strip()

    return response_text

# Iterate over the examples in the validation dataset
for idx, example in enumerate(validation_dataset):
    print(f"\nExample {idx + 1}:\n")
    cv_text = example['cv']
    job_description = example['job_description']
    evaluation = generate_output(cv_text, job_description)

    # Print the generated response
    print("Model Output:")
    print(evaluation)
    print("\n" + "=" * 50 + "\n")