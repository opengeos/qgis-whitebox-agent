"""
Whitebox AI Agent Plugin

An AI-powered agent for running WhiteboxTools through natural language
using Ollama/Claude/OpenAI/Gemini as the reasoning engine.
"""

from .whitebox_agent import WhiteboxAgentPlugin


def classFactory(iface):
    """Load WhiteboxAgentPlugin class.

    Args:
        iface: A QGIS interface instance.

    Returns:
        WhiteboxAgentPlugin: The plugin instance.
    """
    return WhiteboxAgentPlugin(iface)
