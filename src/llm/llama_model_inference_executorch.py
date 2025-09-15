import subprocess
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)


@dataclass
class LlamaParameters:
    seq_len: int = 256
    temperature: float = 0.0
    num_bos: int = 1


class LlamaModelInference:
    def __init__(self,
                 runner_exe: Path,
                 model_path: Path,
                 tokenizer_path: Path,
                 params: Optional[LlamaParameters] = None):
        self.runner_exe = Path(runner_exe)
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.params = params or LlamaParameters()

    def run_inference(self,
                      query: str,
                      system_prompt: Optional[str] = None,
                      max_tokens: int = 200) -> str:
        """Run llama_main binary through subprocess and capture output."""
        if system_prompt:
            prompt = f"{system_prompt}\n{query}"
        else:
            prompt = query

        args = [
            str(self.runner_exe),
            f"--model_path={self.model_path}",
            f"--tokenizer_path={self.tokenizer_path}",
            f"--seq_len={self.params.seq_len}",
            f"--temperature={self.params.temperature}",
            f"--num_bos={self.params.num_bos}",
            f"--prompt={prompt}"
        ]

        logger.info("Launching llama_main...")
        proc = subprocess.run(args,
                              capture_output=True,
                              text=True,
                              check=False)

        if proc.returncode != 0:
            logger.error(proc.stderr)
            raise RuntimeError(f"llama_main failed: {proc.stderr}")

        # llama_main prints the whole generation to stdout
        output_text = proc.stdout.strip()
        logger.info("Generation complete.")
        return output_text

    def stream_inference(self,
                         query: str,
                         system_prompt: Optional[str] = None):
        """Stream tokens from llama_main as they arrive."""
        if system_prompt:
            prompt = f"{system_prompt}\n{query}"
        else:
            prompt = query

        args = [
            str(self.runner_exe),
            f"--model_path={self.model_path}",
            f"--tokenizer_path={self.tokenizer_path}",
            f"--seq_len={self.params.seq_len}",
            f"--temperature={self.params.temperature}",
            f"--num_bos={self.params.num_bos}",
            f"--prompt={prompt}"
        ]

        with subprocess.Popen(args,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True,
                              bufsize=1) as proc:
            for line in proc.stdout:
                yield line.rstrip("\n")
            code = proc.wait()
            if code != 0:
                raise RuntimeError("llama_main exited with code {code}")


if __name__ == "__main__":
    # Example usage
    runner_exe = "/home/derrjohn/executorch/cmake-out/examples/models/llama/llama_main"
    model_path = "/home/derrjohn/Llama3.2-3B-Instruct/llama_3_2-3B-instruct.pte"
    tokenizer_path = "/home/derrjohn/Llama3.2-3B-Instruct/tokenizer.model"

    llama = LlamaModelInference(
        runner_exe=runner_exe,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        params=LlamaParameters(seq_len=500, temperature=0.1)
    )

    # Batch run
    output = llama.run_inference(query="""<|start_header_id|>system<|end_header_id|> 
                                 You are a professional yoga instructor, ONLY provide instructions in THIS json format {'routine_name':'Morning Flow','difficulty':'beginner','total_duration':15,'poses':[{'step':1,'pose_name':'mountain_pose,'hold_duration':30}]. <|eot_id|>
                                 <|start_header_id|>user<|end_header_id|> 
                                 Give me a yoga routine for shoulder tightness.<|eot_id|>
                                 <|start_header_id|>assistant<|end_header_id|>")""")
    print("="*50)
    print(output)

    # Streaming run
    print("="*50)
    for token in llama.stream_inference(query="""<|start_header_id|>system<|end_header_id|> 
                                 You are a professional yoga instructor, ONLY provide instructions in THIS json format {'routine_name':'Morning Flow','difficulty':'beginner','total_duration':15,'poses':[{'step':1,'pose_name':'mountain_pose,'hold_duration':30}]. <|eot_id|>
                                 <|start_header_id|>user<|end_header_id|> 
                                 Give me a yoga routine for lower back.<|eot_id|>
                                 <|start_header_id|>assistant<|end_header_id|>")"""):
        print(token, end="", flush=True)