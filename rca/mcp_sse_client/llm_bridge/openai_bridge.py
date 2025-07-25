"""
OpenAI-specific implementation of the LLM Bridge.
"""
from typing import Dict, List, Any, Optional
import json
import openai
from ..client import ToolDef
from ..format_converters import to_openai_format
from .base import LLMBridge
from .models import DEFAULT_OPENAI_MODEL # Import default model


class OpenAIBridge(LLMBridge):
    """OpenAI-specific implementation of the LLM Bridge."""
    
    def __init__(self, mcp_client, api_key, model=DEFAULT_OPENAI_MODEL): # Use imported default
        """Initialize OpenAI bridge with API key and model.
        
        Args:
            mcp_client: An initialized MCPClient instance
            api_key: OpenAI API key
            model: OpenAI model to use (default: from models.py)
        """
        super().__init__(mcp_client)

        self.llm_client = openai.OpenAI(
            base_url="https://xiaoai.plus/v1",
            api_key=""  # 随便填，Ollama 不校验
        )
        self.model = "gpt-3.5-turbo"


    
    async def format_tools(self, tools: List[ToolDef]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI.
        
        Args:
            tools: List of ToolDef objects
            
        Returns:
            List of tools in OpenAI format
        """
        return to_openai_format(tools)
    
    async def submit_query(self, query: List[Dict[str, Any]], formatted_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit a query to OpenAI with the formatted tools.
        
        Args:
            query: User query string
            formatted_tools: Tools in OpenAI format
            
        Returns:
            OpenAI API response
        """

        
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=query,
            tools=formatted_tools,
            tool_choice="auto"
        )
        
        return response
    
    async def parse_tool_call(self, llm_response: Any) -> Optional[Dict[str, Any]]:
        """Parse the OpenAI response to extract tool calls.
        
        Args:
            llm_response: Response from OpenAI
            
        Returns:
            Dictionary with tool name and parameters, or None if no tool call
        """
        message = llm_response.choices[0].message
        
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            return None
        
        tool_call = message.tool_calls[0]
        
        return {
            "name": tool_call.function.name,
            "parameters": json.loads(tool_call.function.arguments)
        }
