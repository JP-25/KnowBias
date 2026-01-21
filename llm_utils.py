# CACHE_DIR = '/tmp/weights/llama3'
CACHE_DIR = '/scratch/'

import os
# os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
import random
import csv
# import tqdm
import argparse
import torch
import itertools
from transformers import GenerationConfig, pipeline
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import MllamaForConditionalGeneration, MllamaForCausalLM, AutoProcessor, Llama4ForConditionalGeneration, FbgemmFp8Config, AutoModelForImageTextToText

import openai
import time
from typing import *
import json
import re
import copy
import math
import datetime

from torch.nn import functional as F

from cappr.huggingface.classify import cache, predict_proba
from cappr.openai.classify import predict_proba as openai_predict_proba
import numpy as np

from openai import RateLimitError, Timeout, APIError, APIConnectionError, OpenAIError, AzureOpenAI, OpenAI

import base64

import logging

from huggingface_hub import login

from kn_code.patch import unpatch_ff_layers
from kn_code.plot import pre_post_edit_probs, plot
from kn_code import load_model, load_model_train


# Please provide the api key in api_key.txt!
with open("api_key.txt", "r") as f:
    API_KEY = f.readline().strip()
# openai.api_key = API_KEY
 
config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]
login(token = HF_TOKEN)
PROMPT_DIR = 'prompt_instructions'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)



## LLM KN
class LLM_KN:
    
    """LLM wrapper class.
        Meant for use with local Llama-based HuggingFace models (all instruct models). Not using pipleline here
    """

    # temperature=0.3, top_p=0.95, 
    
    def __init__(
        self,
        model_name,                             # Model name to load
        load_in_8bit=False,                     # Option to load in 8-bit (could save memory)
        load_in_4bit=False,
        device_map="auto",                      # Device mapping (GPU by default)
        max_new_tokens=30,                    # Maximum number of new tokens to generate, 2048
        temperature=0.3,                        # Temperature setting for generation
        repetition_penalty=1.2,                 # Penalty for repeating tokens
        top_p=1,                             # Top-p for nucleus sampling
        top_k=50,                               # Top-k tokens considered for generation
        do_sample=True,                        # Whether to use sampling in generation
        cache_dir=CACHE_DIR,    # Directory to cache the weights
        gpu=0,                                  # GPU to use
        verbose=False,                          # Verbosity flag
        train_mode=False,
        # quantization_config=bnb_config,
    ):
        vars = locals() 
        del vars['self']
        for var, value in vars.items():
            setattr(self, var, value)

        
        self.kn = load_model(model_name)

        print("Loaded base LLM here: ", model_name)  
        self.model_name = model_name

    def getOutput(self, prompt, message=None):

        if 'gemma-2' in self.model_name:
            print('load gemma-2, L486')
            m = [
            {
                "role": "user",
                "content": prompt
            }
            ]
        else:
            print('load L494')
            m = [
            {
                "role": "system",
                "content": "You are a useful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
            ]

        if message is not None:
            m = message

        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
        )


        if 'Instruct' in self.model_name or 'chat' in self.model_name:
            return_output = self.kn.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.kn.model.device) ### ori chat template
        else:
            return_output = self.kn.tokenizer(prompt, return_tensors="pt").to(self.kn.model.device)

        output = self.kn.model.generate(**return_output, generation_config=generation_config, pad_token_id=self.kn.tokenizer.eos_token_id)

        if "Llama-2" in self.model_name: # llamma 2 chat model, chat
            response = self.kn.tokenizer.decode(output[0]).split("[/INST]")[1].split("</s>")[0].strip()
        elif "Llama-3" in self.model_name: # 3.1-Instruct prompt format or above
            prompt_length = return_output["input_ids"].shape[1]
            generated_tokens = output[:, prompt_length:]
            if 'Instruct' in self.model_name:
                response = self.kn.tokenizer.decode(generated_tokens[0]).split("<|eot_id|>")[0].strip()
            else:
                response = self.kn.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        elif 'Qwen' in self.model_name or 'gemma' in self.model_name:
            prompt_length = return_output["input_ids"].shape[1]
            generated_tokens = output[:, prompt_length:]
            response = self.kn.tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        if self.verbose:
            print("### RESPONSE ###")
            print(response)
        
        return response

    def getProb(self, prompt, ans, message=None): # test and add up to other LLMs calss
        if 'gemma-2' in self.model_name:
            m = [
            {
                "role": "user",
                "content": prompt
            }
            ]
        else:
            print('load L494')
            m = [
            {
                "role": "system",
                "content": "You are a useful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
            ]

        if message is not None:
            m = message

        self.kn.model.eval()

        def incremental_(prompt, answer):
            prompt_ids = self.kn.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.kn.model.device)
            answer_ids = self.kn.tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)[0]
            
            log_probs_sum = 0.0
            total_tokens = len(answer_ids)

            context_ids = prompt_ids[0]

            for i, token_id in enumerate(answer_ids):
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.kn.model(input_ids)

                    next_token_logits = outputs.logits[0, -1, :]

                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Extract the log probability of the correct next token
                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                # Update the context by adding this token
                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.kn.model.device)], dim=0)

            return log_probs_sum  # return the sum of log probabilities
        
        prob = incremental_(m, ans)
        return math.exp(prob)
    
    def getPerplexity(self, prompt, ans): # test and add up to other LLMs calss
        m = [
        {
            "role": "system",
            "content": "You are a useful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
        ]

        self.kn.model.eval()

        def incremental_perplexity(prompt, answer):
            prompt_ids = self.kn.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.kn.model.device)

            answer_ids = self.kn.tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False)[0]

            log_probs_sum = 0.0
            total_tokens = len(answer_ids)


            context_ids = prompt_ids[0]

            for i, token_id in enumerate(answer_ids):
                input_ids = context_ids.unsqueeze(0)

                with torch.no_grad():
                    outputs = self.kn.model(input_ids)
                    next_token_logits = outputs.logits[0, -1, :]  
                    next_token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
                

                token_log_prob = next_token_log_probs[token_id].item()
                log_probs_sum += token_log_prob

                context_ids = torch.cat([context_ids, torch.tensor([token_id]).to(self.kn.model.device)], dim=0)


            avg_neg_log_likelihood = -log_probs_sum / total_tokens


            perplexity = math.exp(avg_neg_log_likelihood)
            return perplexity, total_tokens
        
        perplexity, total_tokens = incremental_perplexity(m, ans)

        return perplexity, total_tokens