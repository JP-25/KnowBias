import json
import torch
from huggingface_hub import login
from transformers import (
    BertTokenizer, BertLMHeadModel,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, LlamaForCausalLM,
    AutoModelForCausalLM, BitsAndBytesConfig
)
from .knowledge_neurons import KnowledgeNeurons
CACHE_DIR = '/scratch/'
import os
print(os.getcwd())


config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
COHERE_API = config_data["cohere_api"]
login(token = HF_TOKEN)

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased", "bert-base-cased"]
GPT2_MODELS = ["gpt2", "gpt2-xl"]
ALL_MODELS = BERT_MODELS + GPT2_MODELS

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def initialize_model_and_tokenizer(model_name: str, torch_dtype='auto'):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, torch_dtype=torch_dtype)
        model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=torch_dtype)
    elif 'llama' in model_name.lower() or 'Qwen' in model_name or 'gemma' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if 'Vision' in model_name:
            print('vision llama model load')
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True) ## check this
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, cache_dir=CACHE_DIR, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True) ## check this
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval() ### check eval mode

    return model, tokenizer

def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt2"
    elif 'llama' in model_name:
        return 'llama'
    elif 'Qwen' in model_name:
        return 'Qwen'
    elif 'gemma' in model_name:
        return 'gemma'
    else:
        raise ValueError("Model {model_name} not supported")


def load_model(model_name_or_path, device=None):
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(model_name_or_path))
    if kn.model_type in ['gpt2', 'llama', 'Qwen', 'gemma']:
        kn.tokenizer.pad_token = kn.tokenizer.eos_token
    return kn