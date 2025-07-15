import time
import uuid
import sys
from typing import List, Optional, Union, Any, Dict
from pydantic import BaseModel
from pathlib import Path

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


sys.path.append(str(Path(__file__).parent.parent))
from llm.gemma_model_inference import GemmaModelInference
from llm.deepseek_model_inference import DeepSeekModelInference
from model_loader import ModelLoader

app = FastAPI(title="OpenAI Local API")

inference_model = None

@app.on_event("startup")
async def startup_event():
    global inference_model
    iLoad = ModelLoader(model="gemma-3_1b", processor="CPU", model_type="default")
    model_subdirectory = iLoad.model_subdirectory_path
    graphs = iLoad.graphs
    model_sessions = {graph_name: iLoad.load_model(graph, htp_performance_mode="sustained_high_performance") for graph_name, graph in graphs.items() if str(graph).endswith(".onnx")}
    tokenizer = next((file for file in graphs.values() if file.endswith("tokenizer.json")), None)
    meta_data = graphs["META_DATA"]
    print("Loading model...")
    inference_model = GemmaModelInference(
                                    model_sessions=model_sessions,
                                    tokenizer=tokenizer,
                                    model_subdirectory=model_subdirectory,
                                    model_meta=meta_data
                                    )
    print("Model Loaded")

# Not using this at the moment, need to update def startup_event()
MODEL_REGISTRY: Dict[str, Any] = {
    "gemma-1b": GemmaModelInference,
    "deepseek-7b": DeepSeekModelInference
    }

# Request Models
class Message(BaseModel):
    role: str # "system", "user", "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

# Response Models
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    stop: Optional[Union[str, List[str]]] = None

async def call_to_local_llm(messages: List[Message], **kwargs) -> str:
   
    global inference_model
    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature")
    
    prompt = [{"role":msg.role, "content":msg.content} for msg in messages]
    response = inference_model.run_inference(messages=prompt, 
                                             max_tokens=max_tokens,
                                             temperature=temperature)
    
    return response

def estimate_tokens(text: str) -> int:
    """
    Simple token estimation - replace with your tokenizer
    """
    return len(text.split()) * 1.3  # Rough approximation


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Generate unique ID for this completion
    if request.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    model_class = MODEL_REGISTRY[request.model]
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())
        
        # Extract parameters for your LLM
    llm_kwargs = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "stop": request.stop,
            }
    response_content = await call_to_local_llm(request.messages, **llm_kwargs)
    print("\n")
    print("*"*100)
    print(response_content)
    prompt_text = "\n".join([msg.content for msg in request.messages])
    prompt_tokens = int(estimate_tokens(prompt_text))
    completion_tokens = int(estimate_tokens(response_content))

    response = ChatCompletionsResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=response_content),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)