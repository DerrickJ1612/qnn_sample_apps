"""
author: Derrick Johnson
date: 11/07/2025
todo: 
    replace print with logging
    add comments
"""

import onnxruntime as ort
import numpy as np
import logging
import time

from enum import IntEnum, Enum
from tokenizers import Tokenizer
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from utils import apply_repetition_penalty, top_k_probas

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
    num_heads: int=28
    attn_head_size: int=128
    num_layers: int=28
    max_seq_len: int=64
    temp: float=0.6 # Unnecessary, passing directly to generate function from args
    num_key_value_heads: int=4
    seq_len: Optional[int] = None
    hidden_size: Optional[int] = None


logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG in dev
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

class QwenModelInference():
    """
    A class that wraps ONNX inference for DeepSeek language models, including tokenizer initialization
    and model metadata management.

    This class handles loading and managing multiple ONNX Runtime sessions (e.g., embedding, context, head),
    initializes the tokenizer, stores model metadata, and sets up verbosity for debugging or inspection.

    Args:
        model_sessions (Dict[str, ort.InferenceSession]): A mapping of session names (e.g., 'EMBEDDING', 'CONTEXT') 
            to ONNX Runtime InferenceSession objects.
        tokenizer (str): Filename of the tokenizer JSON file located within the model subdirectory.
        model_subdirectory (Path): Path to the model directory containing ONNX files and tokenizer.
        model_meta (dict): A dictionary containing model metadata (e.g., number of layers, heads, etc.).
        verbose (VerbosityLevel, optional): Level of verbosity to control debug output. Defaults to VerbosityLevel.NONE.

    Attributes:
        session_mapper (Dict[str, ort.InferenceSession]): Stores mapped inference sessions.
        tokenizer_path (Path): Full path to the tokenizer file.
        tokenizer (Tokenizer): Initialized tokenizer object.
        model_params (ModelParameters): Parsed and structured model metadata.
        softmax (Callable): Softmax function with temperature scaling for logits.
        verbose (VerbosityLevel): Current verbosity level.
        root_dir (Path): Root working directory at runtime.
    """

    def __init__(self, model_sessions: Dict[str,ort.InferenceSession], 
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

        self.verbosity_init(self.verbose)

    def query(self, query: str, system_prompt: str, persona: Optional[str]=None) -> str:
        """
        UPDATE
        """
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
                        conversation_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|> ")
                        system_prompt = ""
                    else:
                        conversation_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                case "assistant" | "tool":
                    conversation_parts.append(f"<|im_start|>assistant\n{content}<|im_end|> ")
        formatted_query = "\n".join(conversation_parts) + "\n<|im_start|>assistant\n"
        token_ids = self.tokenize(prompt=formatted_query)
        self.model_params.seq_len = token_ids.shape[-1]
        return token_ids

    def _prompt_template(self):
        return """<|im_start|>system
                    [system_instruction]<|im_end|> 
                    <|im_start|>user
                    [query]<|im_end|> 
                    <|im_start|>assistant"""
    
    def tokenize(self, prompt: str) -> np.array:
        """
        Tokenizes the input prompt using the initialized tokenizer.

        This method encodes the prompt into a NumPy array of token IDs, 
        formatted for model input.

        Args:
            prompt (str): The raw input text to tokenize.

        Returns:
            np.array: A 2D NumPy array of shape (1, sequence_length) with dtype int64.
        """
        return np.array([self.tokenizer.encode(prompt).ids], dtype=np.int64)
    
    def embedding_session(self, query: str, persona: Optional[str]=None, iter: bool=True) -> np.array:
        """
        Runs the embedding session to generate token embeddings from a prompt or token IDs.

        If `iter` is False, the method assumes a raw text query and optional persona, 
        constructs the prompt, tokenizes it, and runs the embedding ONNX session.
        If `iter` is True, the method assumes `query` is already a token ID array (used in autoregressive loops).

        This method also updates model parameters for sequence length and hidden size based on the output tensor,
        and handles verbosity during inference.

        Args:
            query (str): The input query string or pre-tokenized token ID array.
            persona (Optional[str]): Optional persona to prepend to the prompt.
            iter (bool): Whether to treat the input as tokenized (`True`) or raw text (`False`).

        Returns:
            np.array: The output from the embedding ONNX session, typically a 3D array of embeddings.
        """
        if not iter:
            prompt = self.query(query, persona)
            token_ids = self.tokenize(prompt)
            verbose = self.verbose
        else:
            # Turn Verbose off within Autoregressive loop
            verbose = 0
            token_ids = query
        expected_outputs = self.session_mapper["EMBEDDING"].get_outputs()[0]
        self.model_params.seq_len = expected_outputs.shape[1]
        self.model_params.hidden_size = expected_outputs.shape[2]

        embedding_output = self.session_mapper["EMBEDDING"].run(None, {"input_ids": token_ids})[0]
        
        self.verbosity_embedding(token_id=token_ids,
                                 embed_output=embedding_output.shape, 
                                 verbose=verbose)
        return embedding_output
    
    def context_session(self, embedding_session_outputs: np.array) -> np.array:
        """
        Runs the context session to generate hidden states and update the KV cache.

        This method initializes the model's key-value cache using outputs from the embedding session,
        executes the context ONNX model, and extracts the hidden states for subsequent layers.
        It also updates internal cache state and optionally logs context-related verbosity details.

        Args:
            embedding_session_outputs (np.array): The output tensor from the embedding session.

        Returns:
            np.array: The context hidden states, typically the first output from the context session.
        """
        init_prompts = self._cache_init(embedding_session_outputs)
        ctx_outputs = self.session_mapper["CONTEXT"].run(None, init_prompts)
        self.kv_cache = self.kv_cache_update(ctx_outputs=ctx_outputs)
        hidden_states = ctx_outputs[0]
        self.verbosity_context(init_prompt_inputs=init_prompts,
                               ctx_outputs=ctx_outputs,
                               verbose=self.verbose)
        return hidden_states

    def head_session(self, ctx_hidden_states: np.array) -> np.array:
        """
        Runs the head session to produce final logits from hidden states.

        This method passes the output hidden states from the context session into the head ONNX model
        and returns the resulting logits, which represent model predictions over the vocabulary.

        Args:
            ctx_hidden_states (np.array): Hidden state tensor from the context session.
                                        

        Returns:
            np.array: The logits tensor, typically of shape.
        """

        logits = self.session_mapper["HEAD"].run(None, {"output_hidden_states": ctx_hidden_states})[0]

        self.verbosity_head(logits=logits,
                            verbose=self.verbose)
        
        return logits
    
    def context_itr_session(self, embedding_session_output: np.array,
                            previous_sequence_length: int=64,
                            io_binding=True):
        """
        Executes a single autoregressive iteration using either IO binding or standard ONNX inference.

        This method performs one step of the iterative decoding process by feeding the current
        embedding output along with past key/value caches and sequence lengths into the context_iter ONNX model.
        Optionally, it uses ONNX Runtime IO binding for improved performance on supported hardware.

        Args:
            embedding_session_output (np.array): Hidden state tensor from the previous layer or token.
            previous_sequence_length (int): Length of tokens already processed; used to compute attention.
            io_binding (bool): If True, uses IO binding for more efficient inference. Otherwise, defaults to standard run.

        Returns:
            np.array: Updated hidden states output from the context iteration.
        
        Raises:
            ValueError: If `io_binding` is enabled but `iBindingManager` is not initialized.
        """
        seq_lengths = {
            "past_seq_len": np.array([[previous_sequence_length]], dtype=np.int32),
            "total_seq_len": np.array([previous_sequence_length+1], dtype=np.int32)
            }
        
        iter_inputs = {
            "input_hidden_states": embedding_session_output,
            **self.kv_cache,
            **seq_lengths
        }
        if io_binding:
            if not hasattr(self,"iBindingManager"):
                    raise ValueError("IO binding cannot proceed: 'iBindingManager' has not been initialized")
            for name, value in iter_inputs.items():
                self.iBindingManager.bind_input(
                    name=name,
                    buffer=value
                )
           
            self.iBindingManager.bind_output(
                name = self.iBindingManager.layer_names[0],
                buffer = self.output_hidden_states_buffer # need to add checks for all used buffers, do it at the top and abstract out
            )
            
            kv_shape_update = (1, self.model_params.num_key_value_heads, previous_sequence_length+1, self.model_params.attn_head_size)
            
            self.iBindingManager.buffer_reallocation_kv(
                updated_buffer_shape=kv_shape_update,
                num_layers=self.model_params.num_layers,
                present_key_buffer=self.present_key_buffer,
                present_value_buffer=self.present_value_buffer,
            )
            # start = time.time()
            self.session_mapper.get("CONTEXT_ITER").run_with_iobinding(self.iBindingManager.io_binding)
            # duration = time.time() - start
            # print("Time per iteration (IOBinding):", duration)
            hidden_states = self.output_hidden_states_buffer
            self.kv_cache = {
                **self.present_key_buffer,
                **self.present_value_buffer
            }
            

        else:
            # start = time.time()
            iter_outputs = self.session_mapper["CONTEXT_ITER"].run(None, iter_inputs)
            # duration = time.time() - start
            # print("Time per iteration (No IOBinding):", duration)
            self.kv_cache = self.kv_cache_update(ctx_outputs=iter_outputs) 
            hidden_states = iter_outputs[0]
        # self.verbosity_context_iter()

        return hidden_states 
        

    def next_token_prediction(self, logits: list, generated_ids: list,
                              temperature: float=1, top_k: Optional[int]=None,
                              repetition_penalty: Optional[float]=None):
        """
        Samples the next token from the output logits using temperature scaling, top-k filtering,
        and optional repetition penalty.

        This method extracts the logits for the last position, applies optional repetition penalties,
        normalizes with softmax, and samples from the distribution. It supports both unrestricted
        sampling and top-k restricted sampling.

        Args:
            logits (list): Logits array of shape (1, seq_len, vocab_size) from the model head.
            generated_ids (list): List of token IDs generated so far (used for repetition penalty).
            temperature (float): Softmax temperature to control randomness (lower = more deterministic).
            top_k (Optional[int]): If provided, restricts sampling to the top-k highest probability tokens.
            repetition_penalty (Optional[float]): If provided, penalizes previously generated tokens.

        Returns:
            int: The ID of the next predicted token.
        """
        last_logit = logits[0,-1]
        if repetition_penalty:
            last_logit = apply_repetition_penalty(logits=last_logit, generated_ids=generated_ids, penalty=repetition_penalty)#self.apply_repetition_penalty(logits=last_logit, generated_ids=generated_ids, penalty=repetition_penalty)

        probas = self.softmax(last_logit, temperature=temperature)
        indices = probas.copy()
 
        if top_k:
            indices, probas = top_k_probas(probas=probas, k=top_k)
            next_token_id = int(np.random.choice(indices, p=probas))
        else:
            next_token_id = int(np.random.choice(len(indices), p=probas))
        
        return next_token_id
    
    def run_inference(self, query: str, 
                      top_k: int, 
                      temperature: float,
                      persona: Optional[str]=None, 
                      max_tokens: int=100,
                      repetition_penalty: float=1.1,
                      io_binding: bool=True
                      ) -> List[str]:
        """
        Runs end-to-end autoregressive inference using a multi-stage ONNX model pipeline.

        This method performs token generation starting from the input query. It initializes the
        embedding and context layers, then iteratively generates tokens using the context iteration
        model (`CONTEXT_ITER`) and head model. Supports ONNX Runtime IOBinding for performance.

        Args:
            query (str): Initial prompt from the user.
            top_k (int): Limits token sampling to top-k most probable choices.
            temperature (float): Sampling temperature; higher values increase randomness.
            persona (Optional[str]): Optional persona name to influence model behavior.
            max_tokens (int): Maximum number of tokens to generate.
            repetition_penalty (float): Penalizes repetition by adjusting logits for previously seen tokens.
            io_binding (bool): If True, uses preallocated buffers and ONNX IOBinding for inference.

        Returns:
            List[int]: A list of generated token IDs, including the first token and any subsequent tokens until
                    either `<|end_of_sentence|>` is reached or `max_tokens` is generated.

        Raises:
            ValueError: If IO binding is enabled but required buffers or manager are not initialized.
        """
        # Reset internal buffers and state
        self.kv_cache = {}
        self.present_key_buffer = {}
        self.present_value_buffer = {}
        self.output_hidden_states_buffer = None

        # Iter set to false because this is prefill stage
        embedding_output = self.embedding_session(query=query, persona=persona, iter=False)
        context_output = self.context_session(embedding_session_outputs=embedding_output)
        logits = self.head_session(ctx_hidden_states=context_output)
        next_token_id = self.next_token_prediction(logits=logits, generated_ids=logits, temperature=temperature)

        generated_ids = [next_token_id]
        prev_sequence_length = self.model_params.max_seq_len

        if io_binding:
            self.iBindingManager = IOBindingManager(inference_session=self.session_mapper["CONTEXT_ITER"])
            _, _, hidden_dimensions = context_output.shape
            _, num_heads, _, head_dimensions = self.kv_cache.get("past_keys_0").shape

            # Initial Buffer allocation
            hidden_state_dimensions = (1,1,hidden_dimensions)
            kv_cache_dimensions = (1, num_heads, prev_sequence_length, head_dimensions)
            self.output_hidden_states_buffer = self.iBindingManager.buffer_preallocation_hidden_states(buffer_shape=hidden_state_dimensions)

            self.present_key_buffer = self.iBindingManager.buffer_preallocation_kv(buffer_shape=kv_cache_dimensions,
                                                                         num_layers=ModelParameters.num_layers,
                                                                         keys_or_values="keys")
            self.present_value_buffer = self.iBindingManager.buffer_preallocation_kv(buffer_shape=kv_cache_dimensions,
                                                                           num_layers=ModelParameters.num_layers,
                                                                           keys_or_values="values")
        logger.info(f"\nInitial Query:\n{query}")
        logger.info("\nGenerated:\n")

        self.verbose = VerbosityLevel.NONE
        for _ in range(max_tokens):
            
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            print(self.tokenizer.decode([next_token_id], skip_special_tokens=True), end="", flush=True)
            
            embedding_output = self.embedding_session(query=input_ids)
            iter_outputs = self.context_itr_session(embedding_session_output=embedding_output,
                                                    previous_sequence_length=prev_sequence_length,
                                                    io_binding=io_binding)
            logits = self.head_session(ctx_hidden_states=iter_outputs)
            next_token_id = self.next_token_prediction(logits=logits, generated_ids=generated_ids,
                                                       temperature=temperature, top_k=top_k,
                                                       repetition_penalty=repetition_penalty)
            generated_ids.append(next_token_id)
            prev_sequence_length += 1

            if next_token_id == self.tokenizer.token_to_id("< | end_of_sentence | >"):
                break

        final_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if io_binding:
            self.iBindingManager.clear_all_bindings()

        return final_response
    
    def kv_cache_update(self, ctx_outputs):
        """
        Updates the key-value (KV) cache based on the output of a transformer model context pass.

        This method extracts past key and value tensors for each transformer layer from the
        context output and stores them in a dictionary with appropriate keys.

        Args:
            ctx_outputs (List[np.ndarray]): The list of output tensors returned by the context session,
                where index 0 is the hidden state and subsequent tensors are interleaved keys and values.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each layer's past keys and values, e.g.,
                {
                    "past_keys_0": ...,
                    "past_values_0": ...,
                    ...
                }
        """    
        present_kv = {f"past_keys_{layer}": ctx_outputs[1 + layer * 2] for layer in range(self.model_params.num_layers)}
        present_kv.update({f"past_values_{layer}": ctx_outputs[1 + layer * 2 + 1] for layer in range(self.model_params.num_layers)})
        return present_kv  
          
    def _build_persona(self, role: InferencePersona) -> str:
        """
        Builds a persona prompt prefix string based on the selected persona role.

        This is used to prepend personality or behavioral instructions to the user query
        during inference to simulate different styles of AI assistant.

        Args:
            role (InferencePersona): An enum representing the desired assistant persona.

        Returns:
            str: A formatted instruction string to prefix the user prompt.
        """
        return f"You are a {role.value}.\n"

    def _cache_init(self, embedding_output: np.array) -> Dict[str,np.array]:
        """
        Initializes an empty KV cache and prepares inputs for the first transformer context pass.

        This method prepares the required input format for the context graph, including:
        - zeroed past key/value caches for each transformer layer,
        - sequence length metadata,
        - and padded input embeddings to match the model's sequence expectations.

        Args:
            embedding_output (np.array): The embedding output array from the embedding session.

        Returns:
            Dict[str, np.array]: A dictionary containing all inputs required for the initial context pass,
                including "past_keys_X", "past_values_X", "input_hidden_states", and sequence length metadata.
        """
        empty_kv = defaultdict()
        output_dimensionality = embedding_output.shape[1]
        past_shape = (self.model_params.batch_size,
                          self.model_params.num_key_value_heads,
                          self.model_params.max_seq_len,
                          self.model_params.attn_head_size)
        
        for layer in range(self.model_params.num_layers):
            empty_kv[f"past_keys_{layer}"] = np.zeros(past_shape, dtype=np.float32)
            empty_kv[f"past_values_{layer}"] = np.zeros(past_shape, dtype=np.float32)
        
        seq_lengths = {
            "past_seq_len": np.array(output_dimensionality-1, dtype=np.int32).reshape(1,1),
            "total_seq_len": np.array([past_shape[2]], dtype=np.int32)
        }
        padded_embedding_outputs = self.input_padding(embedding_output=embedding_output)

        init_prompt_inputs = {
            **empty_kv,
            **seq_lengths,
            "input_hidden_states": padded_embedding_outputs
        }

        return init_prompt_inputs
    

    def input_padding(self, embedding_output, padding_id: int=151643) -> np.array:
        """
        Pads the embedding output to match the model's maximum sequence length.

        Args:
            embedding_output (np.array): The embedding tensor of shape (batch_size, seq_len, embed_dim).
            padding_id (int, optional): Token ID to use for padding (usually a reserved token).

        Returns:
            np.array: A tensor padded along the sequence dimension to (batch_size, max_seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = embedding_output.shape
        padded_embedding = np.full((batch_size, self.model_params.max_seq_len, embed_dim),
                                   padding_id,
                                   dtype=embedding_output.dtype)
        padded_embedding[:, :seq_len, :] = embedding_output
        return padded_embedding
    

    def verbosity_head(self, logits: np.array,
                       verbose: int=VerbosityLevel.NONE):
        match verbose:
            case 1 | 2:
                
                logger.info("\n.....LM Head")
                logger.info(f".....Logits Shape: {logits.shape}")

    def verbosity_context(self, init_prompt_inputs: dict,
                          ctx_outputs: np.array,
                          verbose: int=VerbosityLevel.NONE):
        match verbose:
            case 1 | 2:
                num_layers_verified = (len([layer for layer in init_prompt_inputs.keys() if "past" in layer])-1)//2
                cache_key = next((layer for layer in init_prompt_inputs.keys() if "past_keys" in layer))
                cache_shape = init_prompt_inputs[cache_key].shape
                logger.info("\n.....INITIALIZATION")
                logger.info(f".....Calculated Layers from KV Cache: {num_layers_verified}")
                logger.info(f".....KV Cache Shape: {cache_shape}")
            case _:
                pass

    def verbosity_embedding(self, token_id: List[np.array],
                            embed_output,
                            verbose: int=VerbosityLevel.NONE):
        token_size = token_id.shape[1]
        
        match verbose:
            case 1:
                logger.info(f"\n.....EMBEDDING_SESSION")
                logger.info(f".....Token Count: {token_size}")

            case 2:
                logger.info(f"\n.....EMBEDDING_SESSION")
                logger.info(f".....Token Count: {token_size}")
                logger.info(f".....Prompt:\n{self.tokenizer.decode(token_id.flatten())}")
                logger.info(f".....Token ID: {token_id.flatten()}")
                logger.info(f".....Embedding Output: {embed_output}")

            case _:
                pass 
                
    def verbosity_init(self, verbose: int=VerbosityLevel.NONE):
        model_name = self.model_subdirectory.name
        tokenizer_path = str(self.tokenizer_path)
        
        match verbose:
            case 1:
                logger.info(f"\n.....INIT")
                keys = ", ".join(list(self.session_mapper.keys()))
                logger.info(f".....Model: {model_name}") #logger.info()
                logger.info(f".....Graphs: {keys}")
                logger.info(f".....Tokenizer Path: {tokenizer_path}")

            case 2:
                logger.info(f"\n.....INIT")
                logger.info(f".....Model: {model_name}")
                for graph_name, graph_session in self.session_mapper.items():
                    session_inputs = graph_session.get_inputs()
                    session_outputs = graph_session.get_outputs()
                    input_head = session_inputs[0]
                    output_head = session_outputs[0]
                    logger.info(f".....Graph Name: {graph_name}")
                    logger.info(f".....Expected Input Name: {input_head.name}")
                    logger.info(f".....Expected Input Shape: {input_head.shape}")
                    logger.info(f".....Expected Input Type: {input_head.type}")
                    logger.info("")
                    logger.info(f".....Expected Output Name: {output_head.name}")
                    logger.info(f".....Expected Output Shape: {output_head.shape}")
                    logger.info(f".....Expected Output Type: {output_head.type}")
                    logger.info("."*50)
                logger.info(f".....Tokenizer Path: {tokenizer_path}")

            case _:
                pass

class IOBindingManager():
    def __init__(self, inference_session: ort.InferenceSession):
        self.session = inference_session
        self.outputs = self.session.get_outputs()
        self.layer_names = [output.name for output in self.outputs]
        self.io_binding = self.session.io_binding()

    def buffer_preallocation_hidden_states(self,
                                           buffer_shape: tuple,
                                           dtype: np.dtype=np.float32,
                                        ) -> np.array:
        return np.empty(buffer_shape, dtype=dtype)

    def buffer_preallocation_kv(self, 
                                buffer_shape: Tuple,
                                num_layers: int,
                                keys_or_values: str,
                                dtype: np.dtype=np.float32
                                ) -> Dict[str,str]:
        return {f"past_{keys_or_values}_{layer}": np.empty(buffer_shape, dtype=dtype) \
                for layer in range(num_layers)}   
    
    def buffer_reallocation_kv(self, 
                               updated_buffer_shape: Tuple,
                               num_layers: int,
                               present_key_buffer: Dict[str,str],
                               present_value_buffer: Dict[str,str],
                               dtype: np.dtype=np.float32
                               ) -> None:
        for layer in range(num_layers):
            key_name = self.layer_names[1 + layer * 2]
            value_name = self.layer_names[1 + layer * 2 + 1]

            key_buffer = np.empty(updated_buffer_shape, dtype=dtype)
            value_buffer = np.empty(updated_buffer_shape, dtype=dtype)

            present_key_buffer[f"past_keys_{layer}"] = key_buffer
            present_value_buffer[f"past_values_{layer}"] = value_buffer
            
            self.bind_output(name=key_name,
                             buffer=key_buffer,
                              device_type="cpu",
                              device_id=0,
                              )
            self.bind_output(name=value_name,
                             buffer=value_buffer,
                              device_type="cpu",
                              device_id=0,
                              )
        
    def bind_output(self,
                 name: str,
                 buffer: np.array,
                 device_type: str="cpu",
                 device_id: int=0
                 ) -> None:
        
        self.io_binding.bind_output(
            name=name,
            device_type=device_type,
            device_id=device_id,
            element_type=buffer.dtype,
            shape=buffer.shape,
            buffer_ptr=buffer.ctypes.data)
        
    def bind_input(self,
                    name: str,
                    buffer: np.array,
                    device_id: int=0,
                    device_type: str="cpu",
                    ) -> None:
        
        self.io_binding.bind_input(name,
                                   device_type=device_type,
                                   element_type=buffer.dtype,
                                   shape=buffer.shape,
                                   buffer_ptr=buffer.ctypes.data,
                                   device_id=device_id
                                   ) 
        
    def clear_all_bindings(self):
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()



if __name__=="__main__":
    dummy_dict = {"EMBEDDING":"EMBEDDING_DUMMY",
                  "CONTEXT":"CONTEXT_DUMMY",
                  "CONTEXT_ITER":"CONTEXT_ITER_DUMMY",
                  "HEAD":"HEAD_DUMMY"}
    tokenizer = "tokenizer.json"
    persona = "chef"
    query="provide me a step by step recipe for chicken, be sure to include cook times, ingredients, and preparation"
    model_subdirectory = Path(r"C:\Users\DFS\Desktop\qualcomm_official\Pose-Detection-with-HRPoseNet\models\qnn-deepseek-r1-distill-qwen-7b")
    iInfer = DeepSeekModelInference(dummy_dict, tokenizer, model_subdirectory)
    #query = iInfer.query(query="provide me a step by step recipe for chicken, be sure to include cook times, ingredients, and preparation", persona=persona)
    iInfer.embedding_session(query=query,persona=persona)