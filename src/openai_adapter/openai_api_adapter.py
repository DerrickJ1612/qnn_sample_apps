"""
Figure out why it's not going into tool calls
Oh it is going into selected tools I can see the debug on line 87.
Resolve the cosine similarity. 
The problem is maybe in index tool line 74, the second print on line 94 is printing which returns an
empty list
"""

import time
import uuid
import sys
import json
import numpy as np
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

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("....Warning: sentence-transformers not installed. Tool selection will use fallback method.")
    EMBEDDINGS_AVAILABLE = False

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

class VectorToolSelector:
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.tool_embeddings = {}
        self.tool_descriptions = {}

        if EMBEDDINGS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
    
    def index_tools(self, available_tools: Dict[str, Any]):
        """Pre-compute embeddings for all tool descriptions"""
        self.tool_descriptions = {}

        for tool_name, tool_info in available_tools.items():
            description = tool_info.get("description", "")
            enhanced_description = f"{description} {tool_name.replace('_',' ')}"
            self.tool_descriptions[tool_name] = enhanced_description

            if self.model: 
                embedding = self.model.encode(enhanced_description)
                self.tool_embeddings[tool_name] = embedding
    
    def select_tools(self, query: str, max_tools: int=4) -> List[tuple]:
        """Select tools based on cosine similarity"""
        print(f"DEBUG: {query}")
        if not self.model:
            return self._fallback_selection(query)
        
        if not self.tool_embeddings:
            print("DEBUG: returning some shit")
            return []

        query_embedding = self.model.encode(query)
        similarities = {}

        for tool_name, tool_embedding in self.tool_embeddings.items():
            similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )
            if similarity > self.threshold:
                similarities[tool_name] = similarity
        print(f"DEBUG: {tool_name}: {similarity:.3f} (threshold: {self.threshold})")

        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:max_tools]
    
    def _fallback_selection(self, query: str) -> List[tuple]:
        """Fallback keyword-base selection when embeddings unavailable"""
        query_lower = query.lower()
        matches = []

        keyword_patterns = {
            "identify_yoga_pose":["pose","current","posture","current yoga position","what position"],
            "get_pose_keypoints":["current position","keypoints","position","body position"],
            "get_rubik_sw_info":["software","information","distribution","version","device","os","architecture"]
        }

        for tool_name, keywords in keyword_patterns.items():
            if tool_name in self.tool_descriptions:
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    matches.append((tool_name, min(score/len(keywords), 1.0)))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)

class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.selector = VectorToolSelector()

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], executor_url: str=None):
        """Register a tool with its metadata"""
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "executor_url": executor_url
        }

        self.selector.index_tools(self.tools)

    def get_available_tools(self) -> Dict[str,Any]:
        """Get all available tools"""
        return self.tools
    
    def select_tools_for_query(self, query: str) -> List[str]:
        """Select appropriate tools for a query"""
        selected = self.selector.select_tools(query)
        return [tool_name for tool_name, similarity in selected]

tool_registry = ToolRegistry()

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

# Tool Models for OpenAI compatibility
class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]

# Request Models
class Message(BaseModel):
    role: str # "system", "user", "assistant"
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = "auto"
    #Tool selection mode
    auto_select_tools: Optional[bool] = True
    tool_selection_threshold: Optional[float] = 0.65

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
    selected_tools: Optional[List[str]] = None
    stop: Optional[Union[str, List[str]]] = None

def convert_registry_tools_to_openai_format() -> List[Tool]:
    """Convert internal tool registry to OpenAI Tool format"""
    openai_tools = []
    
    for tool_name, tool_info in tool_registry.get_available_tools().items():
        openai_tool = Tool(
            type="function",
            function=Function(
                name=tool_name,
                description=tool_info["description"],
                parameters=tool_info["parameters"]
            )
        )
        openai_tools.append(openai_tool)
    
    return openai_tools
############################################################################################################

def load_mock_tools():
    """Load some mock tools for testing"""
    mock_tools = {
        "identify_yoga_pose": {
            "description": "Identifies and returns the name of the current yoga pose you're performing. Use when user asks about their current pose, position, or posture.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        "get_rubik_sw_info": {
            "description": "Get system software information including OS distribution, version, and kernel details for the Rubik device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "detail_level": {
                        "type": "string",
                        "enum": ["basic", "full"],
                        "default": "full"
                    }
                }
            }
        },
        "get_weather": {
            "description": "Get current weather information for a specific location including temperature and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        },
        "calculate": {
            "description": "Perform mathematical calculations and solve equations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            }
        }
    }
    
    for tool_name, tool_info in mock_tools.items():
        tool_registry.register_tool(
            name=tool_name,
            description=tool_info["description"],
            parameters=tool_info["parameters"]
        )

async def execute_selected_tools(selected_tools: List[str], query: str) -> Dict[str, Any]:
    """Mock tool execution - for testing"""
    results = {}
    
    for tool_name in selected_tools:
        if tool_name == "identify_yoga_pose":
            results[tool_name] = "cat cow pose"
        elif tool_name == "get_rubik_sw_info":
            results[tool_name] = {
                "device_type": "Rubik Pi",
                "os_info": "Debian",
                "kernel_version": "223314",
                "architecture": "ARM64"
            }
        elif tool_name == "get_weather":
            results[tool_name] = "Unable to determine location from query"
        elif tool_name == "calculate":
            results[tool_name] = "No mathematical expression found in query"
        else:
            results[tool_name] = f"Tool {tool_name} executed"
    
    return results

def create_enhanced_prompt(original_messages: List[Message], tool_results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create enhanced prompt with tool results"""
    prompt = []
    
    # Convert messages to prompt format
    for msg in original_messages:
        prompt.append({"role": msg.role, "content": msg.content or ""})
    
    # Add tool results as context if available
    if tool_results:
        context_parts = []
        for tool_name, result in tool_results.items():
            if isinstance(result, dict):
                result_str = json.dumps(result, indent=2)
            else:
                result_str = str(result)
            context_parts.append(f"{tool_name}: {result_str}")
        
        context = f"\nTool Results:\n{chr(10).join(context_parts)}\n"
        
        # Add context to the last user message or create a system message
        if prompt and prompt[-1]["role"] == "user":
            prompt[-1]["content"] += context
        else:
            prompt.append({"role": "system", "content": f"Additional context:{context}"})
    
    return prompt

###################################################################################################

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
    if not text:
        return 0
    
    return int(len(text.split()) * 1.3)  # Rough approximation


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Generate unique ID for this completion
    if request.model not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
    
    model_class = MODEL_REGISTRY[request.model]
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    created = int(time.time())

    if request.tool_selection_threshold:
        tool_registry.selector.threshold = request.tool_selection_threshold

    selected_tools = {}
    tool_results = {}

    if request.auto_select_tools and request.messages:
        last_user_message = "" 
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                last_user_message = msg.content
                break
        
        if last_user_message:
            selected_tools = tool_registry.select_tools_for_query(last_user_message)

            if selected_tools:
                print(f"Selected tools: {selected_tools}")
                tool_results = await execute_selected_tools(selected_tools, last_user_message)
                print(f"Tool results: {tool_results}")
        
    if tool_results:
        enhanced_messages = create_enhanced_prompt(request.messages, tool_results)
        messages_for_llm = [Message(role=["role"], content=msg["content"]) for msg in enhanced_messages]
    else:
        messages_for_llm = request.messages
        
    # Extract parameters for your LLM
    llm_kwargs = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "stop": request.stop,
            }
    # print(messages_for_llm)
    response_content = await call_to_local_llm(messages_for_llm, **llm_kwargs)
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
        ),
        selected_tools=selected_tools if selected_tools else None
    )
    return response

@app.post("/tools/select")
async def test_tool_selection(query: str):
    """Test tool selection for a query"""
    selected = tool_registry.select_tools_for_query(query)
    return {"query": query, "selected_tools": selected}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# To test
# python -c "import requests; print(requests.get('http://127.0.0.1:8000/tools').json())"