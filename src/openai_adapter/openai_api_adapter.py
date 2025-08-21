from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path
from contextlib import asynccontextmanager
from pprint import pprint
from collections import defaultdict
from sentence_transformers import SentenceTransformer


import uvicorn, sys, time, asyncio, uuid, json
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from llm.gemma_model_inference import GemmaModelInference
from llm.deepseek_model_inference import DeepSeekModelInference
from model_loader import ModelLoader


DEBUG = True
TOOL_THRESHOLD = 0.4

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        iLoad = ModelLoader(model="gemma-3_1b",
                            processor="CPU",
                            model_type="default"
                            )
        graphs = iLoad.graphs

        model_sessions = {
            graph_name: iLoad.load_model(graph, htp_performance_mode="burst")
            for graph_name, graph in graphs.items()
            if str(graph).endswith(".onnx")
        }

        tokenizer = next((file for file in graphs.values() if file.endswith("tokenizer.json")), None)
        meta = graphs["META_DATA"]
        print("Loading Model.....")
        inference_model = GemmaModelInference(
            model_sessions=model_sessions,
            tokenizer=tokenizer,
            model_subdirectory=iLoad.model_subdirectory_path,
            model_meta=meta
        )

        embedding_model = SentenceTransformer("all-mpnet-base-v2")

        app.state.inference = inference_model
        app.state.embedding = embedding_model
        app.state.ready = True

        yield

    finally:
        inf = getattr(app.state, "inference", None)
        if hasattr(inf, "close"):
            try: 
                await asyncio.to_thread(inf.close)
            except Exception:
                pass

app = FastAPI(title="OpenAI Local API", lifespan=lifespan)

def get_inference():
    inf = getattr(app.state, "inference", None)
    if inf is None:
        raise HTTPException(503, "Model not loaded")
    return inf

# ~~~~~ Tool Definition Requests ~~~~~~~
class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef

class ToolChoiceFunction(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any]

ToolChoice = Union[Literal["none", "auto"], ToolChoiceFunction]

# ~~~~~ Tool Definition Response ~~~~~~~
class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction

class Message(BaseModel):
    role: str
    content: Optional[Union[str,None]]
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

# ~~~~~ Request / Response Models ~~~~~~~

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int]=1000
    temperature: Optional[float]=0.3
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None
    

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str]=None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str="chat.completion"
    created: int
    model: str
    choices: List[Choice]


def to_text(x):
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)  # valid JSON string
    return "" if x is None else str(x)

def requires_deterministic_output(messages: List[Message]) -> bool:
    """Check if the task requires more deterministic output (like JSON)."""
    if not messages:
        return False
    
    system_content = (messages[0].content or "").lower()
    deterministic_keywords = ["json", "structured", "format", "schema", "parse"]

    return any(keyword in system_content for keyword in deterministic_keywords)

def _extract_user_query(messages: List[dict]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return str(message.content).strip()
    return ""

def _build_tool_description_map(req_tools) -> Dict[str, str]:
    global DEBUG
    tool_name_description_mapper = defaultdict()
    if not req_tools:
        return tool_name_description_mapper
    
    for tool in req_tools:
        if DEBUG:
            print(tool.function.name)
            print(tool.function.description)
            print("")
            print("*"*100)
        tool_name_description_mapper[tool.function.name] = tool.function.description
    return tool_name_description_mapper

def _vector_process(query: str, request: Request):
    embedding_model = request.app.state.embedding
    return embedding_model.encode(query)

# ~~~~~~~ Move this into utils.py ~~~~~~~~~
def _cosine_similarity(vec_1: np.array, vec_2: np.array) -> float:
    """
    Compute cosine similarity between two vectors.
    Formula: cos(θ) = (A·B) / (||A|| * ||B||)
    """
    dot_product = np.dot(vec_1, vec_2)

    norm_vec_1 = np.linalg.norm(vec_1)
    norm_vec_2 = np.linalg.norm(vec_2)

    if norm_vec_1==0 or norm_vec_2==0:
        return 0.0

    similarity = dot_product / (norm_vec_1 * norm_vec_2)

    return similarity

def _tool_selector(query, tool_descriptions: Dict[str,str], request: Request) -> Optional[str]:
    
    global TOOL_THRESHOLD, DEBUG

    best_score = float("-inf")
    best_tool = None
    query_embed = _vector_process(query=query, request=request)

    for tool_name, description in tool_descriptions.items():
        description_embed = _vector_process(query=description, request=request)

        score = _cosine_similarity(query_embed, description_embed)
        if DEBUG:
            print(f"Tool Name: {tool_name}\nScore: {score}")
        best_score = max(best_score, score)
        if best_score == score:
            best_tool = tool_name
    
    if best_score > TOOL_THRESHOLD:
        return best_tool
    
    return None

async def call_to_local_llm(messages: List[Message], request: Request, tools: bool,**kwargs) -> str:
    """
    Dummy inference function. Replace with your ONNX inference logic.
    For now, just echoes back the input.
    """
    global DEBUG
    # Retrieves instantiated model from FastAPI app object
    inference_engine = request.app.state.inference
    max_tokens = kwargs.get("max_tokens")
    temperature = kwargs.get("temperature")
    top_k = kwargs.get("top_k")

    prompt = [{"role":msg.role, "content":msg.content or ""} for msg in messages]
    if tools:
        prompt = [{"role":messages[-1].role, "content":messages[-1].content}]
    
    if DEBUG:
        print("Temperature Before", temperature, top_k)

    if requires_deterministic_output(messages):
        # Override temperature/top_k to provide semi-deterministic json output
        temperature = 0.2
        top_k = 1

    if DEBUG:  
        print("Temperature After",temperature, top_k)

    def _generate():
        return inference_engine.run_inference(messages=prompt,
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                top_k=top_k)
    
    response = await asyncio.to_thread(_generate)

    return to_text(response)

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, request: Request):
    global DEBUG

    llm_kwargs = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_k": req.top_k
    }

    if req.messages and req.messages[-1].role == "tool":
        output_text = await call_to_local_llm(req.messages, request=request, tools=True, **llm_kwargs)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=output_text),
                    finish_reason="stop"
                )
            ],
        )
        return response

    tool_descriptions = _build_tool_description_map(req.tools)

    if req.tools and req.tool_choice not in (None, "none"):
        query = _extract_user_query(req.messages)

        is_tool = _tool_selector(query=query, tool_descriptions=tool_descriptions, request=request)

        if is_tool:
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created = int(time.time())

            assistant_msg = Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call_id, 
                        function=ToolCallFunction(name=is_tool, arguments="{}")
                        )
                    ],
            )

            response = ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=assistant_msg,
                    finish_reason="tool_calls"
                )
            ],
            )
            return response

    output_text = await call_to_local_llm(req.messages, 
                                          request=request, 
                                          tools=req.tools,
                                          **llm_kwargs)
    # output_text = "for dev only"
    if DEBUG:
        print(output_text, type(output_text))

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=output_text),
                finish_reason="stop"
            )
        ],
    )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)