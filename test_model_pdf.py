import os
import sys
import json
import torch
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

torch.set_num_threads(8)

def convert_pdf_to_text(pdf_path):
    # Extract text from PDF
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    cv_text = text.strip()
    
    # Define the JSONL format output
    output = {
        "instruction": "Evaluate the following CV against the job description and provide a structured response. Your evaluation should include: a match percentage (from 0-100%), good impressions, any warnings, and bad impressions (if applicable).",
        "job_description": "We are looking for a Python Developer with 3+ years of experience in Django, Flask, and REST APIs.",
        "cv": cv_text,  # Insert the extracted CV text here
        "response": {
            "match_percentage": 0,
            "good_impression": None,
            "warning": None,
            "bad_impression": None
        }
    }
    
    # Convert to JSON string and return
    return json.dumps(output)

def evaluate_cv(cv_text, job_description, model, tokenizer, device):
    prompt = f"""### Instruction:
Evaluate the following CV against the job description and provide a structured response. Your evaluation should include:
- A match percentage (from 0-100%)
- Good impressions
- Any warnings
- Bad impressions (if applicable)

### Job Description:
{job_description}

### CV:
{cv_text}

### Response:
"""
    # Tokenize and move inputs to the device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,  # Adjust as needed
            do_sample=False,
            top_p=0.9,        # Adjust as needed
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the response after "### Response:"
    response_text = generated_text.split("### Response:")[-1].strip()

    # Try to parse the response as JSON
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError:
        print("Warning: The model's response is not valid JSON. Outputting raw text.")
        response = response_text

    return response

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate_cv.py path_to_cv.pdf")
        sys.exit(1)

    cv_pdf_path = sys.argv[1]

    if not os.path.exists(cv_pdf_path):
        print(f"The file '{cv_pdf_path}' does not exist.")
        sys.exit(1)

    # Convert the PDF CV to JSONL formatted text
    try:
        cv_jsonl = convert_pdf_to_text(cv_pdf_path)
        if not cv_jsonl:
            print("Error: No text could be extracted from the PDF.")
            sys.exit(1)
    except Exception as e:
        print(f"Error converting PDF to JSONL format: {e}")
        sys.exit(1)

    # Parse JSONL for job_description and cv text
    parsed_json = json.loads(cv_jsonl)
    job_description = parsed_json["job_description"]
    cv_text = parsed_json["cv"]

    # Paths to the base model and fine-tuned model
    base_model_path = "/Users/oktayosman/Downloads/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"  # Update this path as needed
    fine_tuned_model_path = "/Users/oktayosman/Downloads/fine-tuned-model"  # Update this path as needed

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True, trust_remote_code=True)
        # Load the fine-tuned model
        model = PeftModel.from_pretrained(base_model, fine_tuned_model_path, local_files_only=True)
        model.to(device, dtype=torch.bfloat16)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    # Evaluate the CV using the parsed job description and CV text
    evaluation = evaluate_cv(cv_text, job_description, model, tokenizer, device)

    # Output the evaluation
    print("Evaluation Result:")
    if isinstance(evaluation, dict):
        print(json.dumps(evaluation, indent=2))
    else:
        print(evaluation)

if __name__ == "__main__":
    main()
