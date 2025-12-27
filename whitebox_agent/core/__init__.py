"""
Whitebox AI Agent - Core Module

Contains the core components for the AI agent:
- Processing registry adapter for algorithm discovery
- Context builder for LLM prompts
- LLM client for multiple backends
- Agent executor for action handling
"""

from .processing_registry import ProcessingRegistryAdapter
from .context_builder import ContextBuilder
from .llm_client import LLMClient
from .agent_executor import AgentExecutor

__all__ = [
    "ProcessingRegistryAdapter",
    "ContextBuilder",
    "LLMClient",
    "AgentExecutor",
]
