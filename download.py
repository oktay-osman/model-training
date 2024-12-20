from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-3.2-3B-Instruct"
custom_directory = "/Users/oktayosman/hugging-face-models"

# Download and load model and tokenizer into a custom directory
model = AutoModel.from_pretrained(model_name, cache_dir=custom_directory)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_directory)