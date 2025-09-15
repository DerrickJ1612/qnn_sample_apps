from transformers import AutoTokenizer
from pathlib import Path



tok = AutoTokenizer.from_pretrained("/home/derrjohn/Llama3.2-1B-Instruct/")

tok.save_pretrained("/home/derrjohn/Llama3.2-1B-Instruct/converted")