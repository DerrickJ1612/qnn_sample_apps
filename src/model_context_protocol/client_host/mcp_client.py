# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the conditions in LICENSE.txt are met

import asyncio
import json
import os
import sys
import re
from contextlib import AsyncExitStack
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

# Add LLM path to sys.path
file_root = Path.cwd().parent.parent
path_2_llm = file_root / "gitrepo" / "qnn_sample_apps" / "src"
sys.path.append(str(path_2_llm))



class SecurityLevel(Enum):
    """Security levels for LLM providers"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Constants
DEFAULT_SECURITY_LEVEL = SecurityLevel.LOW
DEFAULT_MCP_SERVER_URL = "http://localhost:3001/mcp"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.6 #0.9 wasn't deterministic enough
# SYSTEM_PROMPT = "You are a very experienced Yoga instructor. Be helpful, knowledgeable, and encouraging."
SYSTEM_PROMPT = """
                    You are a yoga assistant. 
                    POSE CHECKING: When using get_current_pose, compare expected_pose vs predicted_pose. Use is_correct_pose field for feedback.
                    Correct: Give positive feedback  
                    Incorrect: Provide specific corrections
                    DO NOT CALL TOOLS WHEN ASKING FOR YOGA ROUTINE 
                """

# LLM Provider configurations
LLM_PROVIDERS = {
    SecurityLevel.LOW: {
        "name": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "model": "claude-3-haiku-20240307",
        "description": "Anthropic Claude for general queries"
    },
    SecurityLevel.MEDIUM: {
        "name": "ai100",
        "api_key_env": "CIRRASCALE_API_KEY",
        "base_url_env": "https://aisuite.cirrascale.com/apis/v2",
        "base_url": "https://aisuite.cirrascale.com/apis/v2",
        "model": "Llama-3.1-8B",
        "description": "AI100 for business queries"
    },
    SecurityLevel.HIGH: {
        "name": "snapdragon",
        "api_key": "local-key",
        "base_url_env": "LOCAL_LLM_BASE_URL",
        "base_url": "http://localhost:8000/",
        "model": "gemma-1b",
        "description": "Local Snapdragon for high-security queries"
    }
}

# Security level detection keywords
SECURITY_KEYWORDS = {
    SecurityLevel.HIGH: [
        "personal", "private", "confidential", "medical", "health",
        "password", "account", "financial", "bank", "injury", "pain"
    ],
    SecurityLevel.MEDIUM: [
        "business", "company", "work", "professional", "client",
        "meeting", "project", "strategy", "revenue"
    ]
}


class LLMClientManager:
    """Manages LLM client creation and configuration"""
    
    @staticmethod
    def create_client(security_level: SecurityLevel) -> Tuple[Any, str]:
        """Create and return an OpenAI-compatible client for the specified security level"""
        from openai import OpenAI
        
        load_dotenv()
        
        config = LLM_PROVIDERS[security_level]
        
        # Get API key
        if config["name"] == "snapdragon":
            api_key = config["api_key"]
        else:
            api_key = os.getenv(config["api_key_env"])
            if not api_key:
                raise ValueError(f"{config['api_key_env']} environment variable is required for {config['name']}")
        
        # Get base URL
        base_url = os.getenv(config.get("base_url_env", ""), config["base_url"])
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        return client, config["model"]


class SecurityAnalyzer:
    """Analyzes queries to determine appropriate security level"""
    
    @staticmethod
    def determine_security_level(query: str) -> SecurityLevel:
        """Analyze query to determine appropriate security level"""
        query_lower = query.lower()
        
        # Check for high security keywords first
        if any(keyword in query_lower for keyword in SECURITY_KEYWORDS[SecurityLevel.HIGH]):
            return SecurityLevel.HIGH
        
        # Check for medium security keywords
        if any(keyword in query_lower for keyword in SECURITY_KEYWORDS[SecurityLevel.MEDIUM]):
            return SecurityLevel.MEDIUM
        
        return SecurityLevel.LOW


class MCPHandler:
    """Handles MCP server communication"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.stream_ctx = None
        self.session_ctx = None
    
    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            self.stream_ctx = streamablehttp_client(self.server_url)
            self.read_stream, self.write_stream, self.get_session_id = await self.stream_ctx.__aenter__()
            
            self.session_ctx = ClientSession(
                read_stream=self.read_stream,
                write_stream=self.write_stream
            )
            
            self.session = await self.session_ctx.__aenter__()
            await self.session.initialize()
            print(" Connected to MCP server")
            return True
        except Exception as e:
            print(f" Failed to connect to server: {e}")
            return False
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from MCP server"""
        if not self.session:
            return []
        
        try:
            response = await self.session.list_tools()
            return [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools]
        except Exception as e:
            print(f" Error getting tools: {e}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Call a tool from MCP server"""
        if not self.session:
            return None
        
        try:
            return await self.session.call_tool(name=tool_name, arguments=arguments)
        except Exception as e:
            print(f" Error calling tool {tool_name}: {e}")
            return None
    
    async def close(self):
        """Clean up MCP connection resources"""
        try:
            if self.session and self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if self.stream_ctx:
                await self.stream_ctx.__aexit__(None, None, None)
        except Exception as e:
            print(f"Warning: Error during MCP cleanup: {e}")


class ConversationManager:
    """Manages conversation history and message processing"""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to conversation history"""
        message = {"role": role, "content": content}
        message.update(kwargs)
        self.history.append(message)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages formatted for LLM"""
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.history
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()
    
    def __len__(self) -> int:
        return len(self.history)


class YogaChatBot:
    """Main chat bot class"""
    
    def __init__(self, server_url: str = DEFAULT_MCP_SERVER_URL):
        self.mcp_handler = MCPHandler(server_url)
        self.conversation = ConversationManager()
        self.cache_path = Path("__file__").resolve().parent/"__mcp_cache__"
        self.security_level = DEFAULT_SECURITY_LEVEL
        self.current_llm_client = None
        self.current_model = None
        self.manual_security_override = False  # Track if user manually set security level
        
        # Initialize with default security level
        self._setup_llm_client(DEFAULT_SECURITY_LEVEL)
    
    def _setup_llm_client(self, security_level: SecurityLevel):
        """Setup LLM client for the specified security level"""
        try:
            self.current_llm_client, self.current_model = LLMClientManager.create_client(security_level)
            self.security_level = security_level
            self.conversation.clear()
            print(f"Switched to {security_level.value} security level ({LLM_PROVIDERS[security_level]['description']})")
        except Exception as e:
            print(f" Error setting up {security_level.value} client: {e}")
            print("💡 Please check your API keys and configuration")
    
    def _create_llm_response(self, messages: List[Dict[str, str]], available_tools: Optional[List] = None, is_routine: Optional[bool]=None):
        """Create LLM response using current client and model"""
        api_params = {
            "model": self.current_model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE
        }

        if available_tools and not is_routine:
            api_params["tools"] = available_tools 
            api_params["tool_choice"] = "auto"
        
        return self.current_llm_client.chat.completions.create(**api_params)
    
    async def _handle_tool_calls(self, message, available_tools: List[Dict[str, Any]]) -> str:
        """Handle tool calls from LLM response"""
        print("Using tools...")
        
        # Add assistant message to history
        self.conversation.add_message(
            role="assistant",
            content=message.content,
            tool_calls=message.tool_calls
        )
        
        # Process tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            
            if tool_name == "begin_routine":
                cached_routine = self._load_cache_routine() 
                # tool_args["complete_routine"] = cached_routine
                tool_args = {"complete_routine": cached_routine}

            print(f"Calling {tool_name}...")
            result = await self.mcp_handler.call_tool(tool_name, tool_args)

            if result:
                content_str = self._extract_tool_result_content(result)
                self.conversation.add_message(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=content_str
                )
        
        # Get final response after tool calls
        messages = self.conversation.get_messages_for_llm()
        final_response = self._create_llm_response(messages, available_tools)
        return final_response.choices[0].message.content
    
    def _extract_tool_result_content(self, result) -> str:
        """Extract content from tool result"""
        if isinstance(result.content, list):
            content_str = ""
            for item in result.content:
                if hasattr(item, 'text'):
                    content_str += item.text
                else:
                    content_str += str(item)
            return content_str
        return str(result.content)

    def _extract_json_from_response(self, llm_response: str):
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        match = re.search(json_pattern, llm_response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    
        return None

    def _cache_routine(self, llm_response: str):

        try:
            routine_json = self._extract_json_from_response(llm_response)

            if self._validate_routine_structure(routine_json=routine_json):
                with open(str(self.cache_path.joinpath("routine_cache.json")),"w") as file:
                    json.dump(routine_json, file, indent=2)
            return True
        except json.JSONDecodeError:
            print("Not correct JSON Format")
            return False
        except Exception as e:
            print(f"Failed to cache routine: {e}")
            return False
        
    def _load_cache_routine(self) -> Dict:
        try: 
            with open(self.cache_path.joinpath("routine_cache.json"),"r") as file:
                return json.load(file)
        except FileNotFoundError:
            print("Cache Empty")
            return {}
    
    def _validate_routine_structure(self, routine_json: dict) -> bool:
        required_fields = ["routine_name", "poses"]
        required_pose_fields = ["pose_name","instructions","hold_duration"]

        if not all(field in routine_json for field in required_fields):
            return False

        if not isinstance(routine_json["poses"], list):
            return False
        
        for idx, pose in enumerate(routine_json["poses"]):
            missing_fields = [field for field in required_pose_fields if field not in pose]
            if missing_fields:
                return False
        return True
    
    def _is_valid_tool_call(self, message):
        """
        Validate that a tool call is properly formed and callalble.

        Need this check because Medium Security Host (Cirrascale) is always returning a tool ID even when there are no tool calls.
        This is unexpected behavior but results in Medium Security trying to call a non-existent tool
        """
        try:
            return bool(message.tool_calls.function.name)
        except AttributeError:
            return False

    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and return response"""
        # Auto-detect security level only if not manually overridden
        if not self.manual_security_override:
            suggested_level = SecurityAnalyzer.determine_security_level(user_message)
            if suggested_level != self.security_level:
                print(f"Detected {suggested_level.value} security level for this query")
                self._setup_llm_client(suggested_level)
        
        # Add user message to conversation history
        self.conversation.add_message("user", user_message)
        
        # Prepare messages for LLM
        messages = self.conversation.get_messages_for_llm()
        
        # Update to handle routine requests that don't require tools,
        # need very strict JSON format for routine walkthrough
        routine_keywords = ["yoga routine", "yoga sequence", "yoga flow", 
                   "yoga practice", "yoga workout", "yoga session",
                   "generate a routine", "create a routine", "make a routine"]
        is_routine_request = any(keyword in user_message.lower() for keyword in routine_keywords)

        if is_routine_request:
            # print("Routine Request")
            json_format = {
            "routine_name": "Morning Flow",
            "difficulty": "beginner", 
            "total_duration": 15,
            "poses": [
                {
                "step": 1,
                "pose_name": "Neck Rolls",
                "hold_duration": 60,
                "instructions": "Sit comfortably, place hands on your knees. Gently roll your head from side to side, looking as far as you comfortably can.  Focus on relaxing your neck muscles. (30 seconds)",
                "key_alignments": [
                    "relaxed shoulders"
                ]},
                {
                "step": 2,
                "pose_name": "Chin Tucks",
                "hold_duration": 60,
                "instructions": "Sit tall with a straight spine. Gently tuck your chin towards your chest, holding for 5-10 seconds. Repeat 5-10 times. (30 seconds)",
                "key_alignments": [
                    "relaxed shoulders"
                ]}
                ]
            }
            routine_system_prompt = f"""
            Generate yoga routine JSON. Output ONLY valid JSON, no other text. 5 steps max!

            THIS IS THE JSON FORMAT: 
            <begin json format>
            {json.dumps(json_format)}
            <end json format>
            
            Create routine now using the JSON format as an example.
            Must include: hold_duration, key_alignments. 
            DO NOT INCLUDE ANYTHING OTHER THAN THE JSON FORMAT!
            """
            messages = [{"role":"system", "content": routine_system_prompt}] + self.conversation.history
            available_tools = []
        
        else:
            messages = self.conversation.get_messages_for_llm()
            available_tools = await self.mcp_handler.get_available_tools()
  
        try:
            # Make initial LLM call
            response = self._create_llm_response(messages, available_tools, is_routine_request)
            message = response.choices[0].message
            response_text = message.content or ""
            
            if hasattr(message, 'tool_calls') and message.tool_calls: 
                response_text = await self._handle_tool_calls(message, available_tools)

            self.conversation.add_message("assistant", response_text)
            if is_routine_request:
                
                self._cache_routine(llm_response=response_text)

            return response_text
        
        except Exception as e:
            error_msg = f" Error processing message: {e}"
            print(error_msg)
            return error_msg
        
    async def _handle_command(self, command: str) -> bool:
        command = command.lower()

        if command == '/clear':
            self.conversation.clear()
            print("Conversation history cleared!")
        elif command == '/low':
            self._setup_llm_client(SecurityLevel.LOW)
            self.manual_security_override = True
            print("Manual security level override enabled. Use /auto to re-enable auto-detection.")
        elif command == '/medium':
            self._setup_llm_client(SecurityLevel.MEDIUM)
            self.manual_security_override = True
            print("Manual security level override enabled. Use /auto to re-enable auto-detection.")
        elif command == '/high':
            self._setup_llm_client(SecurityLevel.HIGH)
            self.manual_security_override = True
            print("Manual security level override enabled. Use /auto to re-enable auto-detection.")
        elif command == '/auto':
            self.manual_security_override = False
            print("Auto-detection re-enabled. Security level will be determined automatically.")
        elif command == '/bye':
            print("Thanks for chatting, talk to you next time")
            return False
        
        else:
            print("Unknown command.")
        
        return True
    
    async def start_chat(self):
        print("Welcome to Yoga Chat Assistant!")
        print("=" * 50)
        
        # Try to connect to MCP server
        await self.mcp_handler.connect()
        
        while True:
            try:
                user_input = input(f"\n[{self.security_level.value}] You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    should_continue = await self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue
                
                # Process regular message
                print(f"\n[{self.security_level.value}] Assistant: ", end="", flush=True)
                response = await self.process_message(user_input)
                print("\n", response)
                
            except Exception as e:
                print(f"\n Error: {e}")
    
    async def close(self):
        """Clean up resources"""
        await self.mcp_handler.close()


def check_available_providers():
    """Check and display available LLM providers"""
    print("Checking available providers...")
    for level, config in LLM_PROVIDERS.items():
        if config.get("api_key_env"):
            api_key = os.getenv(config["api_key_env"])
            status = "GOOD" if api_key else "BAD"
            print(f"  {level.value}: {status} {config['description']}")
        else:
            print(f"  {level.value}: {config['description']}")



async def main():
    """Main function to run the chat application"""
    load_dotenv()
    
    check_available_providers()
    
    # Create and start chat bot
    chat_bot = YogaChatBot()
    
    try:
        await chat_bot.start_chat()
    except Exception as e:
        print(f" Error: {e}")
    finally:
        await chat_bot.close()


if __name__ == "__main__":
    asyncio.run(main())