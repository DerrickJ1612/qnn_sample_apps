"""
author: Derrick Johnson
date: 05/12/2025
todo: 
    replace print with logging
    add comments
"""

import onnxruntime as ort
import numpy as np
import logging
import time
import re
import sys

from enum import IntEnum, Enum
from tokenizers import Tokenizer
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from utils import apply_repetition_penalty, top_k_probas

# Move into a utils
class VerbosityLevel(IntEnum):
    NONE = 0
    BASIC = 1
    DETAILED = 2

class InferencePersona(Enum):
    THERAPIST = "therapist"
    CYBER_SECURITY ="cyber security specialist"
    CHEF ="chef"
    CARE_TAKER = "care taker"
    DOCTOR = "medical professional"

@dataclass
class ModelParameters:
    batch_size: int=1
    num_heads: int=12
    attn_head_size: int=128
    num_layers: int=28
    max_seq_len: int=64
    temp: float=0.6 # Unnecessary, passing directly to generate function from args
    num_key_value_heads: int=2
    seq_len: Optional[int] = None
    hidden_size: Optional[int] = None

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG in dev
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

class GemmaModelInference():
    
    def __init__(self, model_sessions: Dict[str, ort.InferenceSession],
                 tokenizer: str,
                 model_subdirectory: Path,
                 model_meta: dict,
                 verbose: VerbosityLevel = VerbosityLevel.NONE):
        self.session_mapper = model_sessions
        self.root_dir = Path.cwd()
        self.model_subdirectory = model_subdirectory
        self.tokenizer_path = model_subdirectory/tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.model_params = ModelParameters(**model_meta)
        self.verbose = verbose
        self.softmax = lambda x, temperature=1: np.exp((x-np.max(x))/temperature)/np.sum(np.exp((x-np.max(x))/temperature), axis=-1)

        # self.verbosity_init(self.verbose)
    
    def query(self, query: str, system_prompt: Optional[str]=None, persona: Optional[str]=None) -> str:
        
        self.query = query

        if system_prompt:
            formatted_query = self._prompt_template().replace("[system_instruction]",system_prompt)\
                                                     .replace("[query]",query)
        else:
            formatted_query = self._prompt_template().replace("[system_instruction].","")\
                                                     .replace("[query]",query)
        
        token_ids = self.tokenize(prompt=formatted_query)
        self.model_params.seq_len = token_ids.shape[-1]
        return token_ids
    
    def query_builder(self, messages: List[Dict[str,str]]) -> str:
        self.messages = messages
        conversation_parts = []
        system_prompt = ""
        for message in self.messages:
            content = message["content"]
            match message["role"]:
                case "system":
                    system_prompt = content
                case "user":
                    if system_prompt:
                        conversation_parts.append(f"<start_of_turn>user\n{system_prompt}\n{content}<end_of_turn>")
                        system_prompt = ""
                    else:
                        conversation_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                case "assistant":
                    conversation_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        formatted_query = "\n".join(conversation_parts) + "\n<start_of_turn>model\n"
        token_ids = self.tokenize(prompt=formatted_query)
        self.model_params.seq_len = token_ids.shape[-1]
        return token_ids

    def tokenize(self, prompt: str) -> np.array:
        return np.array([self.tokenizer.encode(prompt).ids], dtype=np.int64)

    def _cache_init(self):
        empty_kv = defaultdict()
        past_shape = (self.model_params.batch_size,
                          self.model_params.num_key_value_heads,
                          self.model_params.seq_len,
                          self.model_params.attn_head_size
                          )
        
        for layer in range(self.model_params.num_layers):
            empty_kv[f"past_key_values.{layer}.key"] = np.zeros(past_shape, dtype=np.float32)

            empty_kv[f"past_key_values.{layer}.value"] = np.zeros(past_shape, dtype=np.float32)
        
        return empty_kv
    
    def _position_ids(self, sequence_id: List):
          return np.array([sequence_id], dtype=np.int64)
    
    def _attention_mask(self, seq_len: int):
        return np.ones((self.model_params.batch_size, seq_len), dtype=np.int64)
    
    def _input_process(self, input_ids: np.array,
                            current_seq_length: int,
                            position_ids: np.array,
                            kv_cache: Dict[str,float]) -> Dict[str,any]:
        return {"input_ids": input_ids,
                "attention_mask": self._attention_mask(seq_len=current_seq_length),
                "position_ids": self._position_ids(sequence_id=position_ids),
                **kv_cache}
    
    def kv_cache_update(self, model_outputs):
        present_kv = {f"past_key_values.{i}.key": model_outputs[1 + i * 2] for i in range(self.model_params.num_layers)}
        present_kv.update({f"past_key_values.{i}.value": model_outputs[1 + i * 2 + 1] for i in range(self.model_params.num_layers)})
        return present_kv
    
    def next_token(self, model_outputs: Dict, temperature: float, top_k: Optional[int]=None):
        logits = model_outputs[0]
        probas = self.softmax(logits[0,-1], temperature=temperature)
        indices = probas.copy()
        if top_k:
            indices, probas = top_k_probas(probas=probas, k=top_k)
            next_token_id = int(np.random.choice(indices, p=probas))
        else:
            next_token_id = int(np.random.choice(len(probas), p=probas))
        return next_token_id
    
    def decode(self, input_ids: np.array, next_token_id: int, kv_cache: Dict, max_tokens: int,
               temperature: float, top_k: int):
        generated_ids = [next_token_id]
        sequence_length = input_ids.shape[-1]
        position_iter = input_ids.shape[-1]
        printed_length = 0

        logger.info(f"\nInitial Query:\n{self.query}")
        logger.info("\nGenerated:\n")

        for _ in range(max_tokens):
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            position_ids = np.array([[position_iter]], dtype=np.int64)
            input_dict = self._input_process(input_ids=input_ids,
                                             current_seq_length=sequence_length,
                                             position_ids=[position_iter],
                                             kv_cache=kv_cache)
            decode_output = self.session_mapper["MODEL"].run(None, input_dict)
            sequence_length += 1
            position_iter += 1

            kv_cache = self.kv_cache_update(decode_output)
            next_token_id = self.next_token(model_outputs=decode_output,
                                            temperature=temperature,
                                            top_k=top_k)
            generated_ids.append(next_token_id)
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            new_text = full_text[printed_length:]
            if new_text:
                print(new_text, end="", flush=True)
                printed_length = len(full_text)

            if next_token_id == self.tokenizer.token_to_id("<end_of_turn>"):
                break    

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text
    
    def _prompt_template(self):
        return """<start_of_turn>user
                    [system_instruction]. [query]
                    <end_of_turn>
                    <start_of_turn>model"""
    
    def run_inference(self, 
                      query: Optional[str]=None,
                      messages: Optional[List[Dict[str,str]]]=None,
                      top_k: int=50,
                      temperature: float=0.6,
                      persona: Optional[str]=None,
                      max_tokens: int=100,
                      system_prompt: Optional[str]=None,
                      io_binding: Optional[bool]=None,
                      repetition_penalty: Optional[float]=None) -> List[str]:
        
        if query is None and messages is None:
            raise ValueError("Either 'query' or 'query_builder' must be provided")
        if query: 
            token_ids = self.query(query=query, system_prompt=system_prompt)
        if messages:
            token_ids = self.query_builder(messages=messages)

        kv_cache = self._cache_init()
        len_token_id = token_ids.shape[-1]
        position_ids = np.array(np.arange(len_token_id, dtype=np.int64))
        input_dict = self._input_process(input_ids=token_ids,
                                         current_seq_length=len_token_id,
                                         position_ids = position_ids,
                                         kv_cache=kv_cache)
        prefill_output = self.session_mapper["MODEL"].run(None, input_dict)
        kv_cache = self.kv_cache_update(model_outputs=prefill_output)
        next_token_id = self.next_token(prefill_output, temperature=temperature, top_k=top_k)

        decode_output = self.decode(input_ids=token_ids,
                                    next_token_id=next_token_id,
                                    kv_cache=kv_cache,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    top_k=top_k)
        print("*"*50)
        print(decode_output)
        return decode_output
        # print(self.tokenizer.decode([decode_output]))
        
if __name__=="__main__":
    import sys
    sys.path.append(str(Path.cwd()/"src"))
    from model_loader import ModelLoader

    model_name = "GEMMA-3_1B"
    processor = "cpu"
    model_type = "default"
    iLoad = ModelLoader(model=model_name, processor=processor, model_type=model_type)
    model_subdirectory = iLoad.model_subdirectory_path

    graphs = iLoad.graphs
    model_sessions = {graph_name: iLoad.load_model(graph,htp_performance_mode="sustained_high_performance") for graph_name,graph in graphs.items() if str(graph).endswith(".onnx")}
    tokenizer = next((file for file in graphs.values() if file.endswith("tokenizer.json")), None)
    meta_data = graphs["META_DATA"]
    
    iInfer = GemmaModelInference(
        model_sessions=model_sessions,
        tokenizer=tokenizer,
        model_subdirectory=model_subdirectory,
        model_meta=meta_data
        )
    iInfer.run_inference(query="Why is some snow better than other snow", system_prompt="you are a professional skier", max_tokens=500)

