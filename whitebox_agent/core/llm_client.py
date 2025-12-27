"""
LLM Client

Unified client for multiple LLM backends:
- Ollama (local)
- Claude (Anthropic)
- OpenAI
- Gemini (Google)
"""

import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK = "bedrock"


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: LLMProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096


# System prompt for the Whitebox AI agent
SYSTEM_PROMPT = """You are an AI assistant that executes WhiteboxTools geospatial algorithms through QGIS.

## CRITICAL RULES
1. ALWAYS respond with a valid JSON object - no plain text ever
2. Read LAYER_CATALOG carefully and use the EXACT "source:" paths shown there
3. For multi-step workflows, execute ONE step at a time

## ALGORITHM REFERENCE
- FillDepressions: wbt:FillDepressions (dem, output)
- FlowAccumulation: wbt:D8FlowAccumulation (input, output) - NOTE: param is "input"
- FlowDirection: wbt:D8Pointer (dem, output) 
- ExtractStreams: wbt:ExtractStreams (flow_accum, threshold, output) - REQUIRES flow accumulation raster!
- Slope: wbt:Slope (dem, output)
- Aspect: wbt:Aspect (dem, output)
- Hillshade: wbt:Hillshade (dem, output)

## MULTI-STEP WORKFLOWS
Some operations require intermediate steps:
- "Extract streams from DEM" requires: 1) Run D8FlowAccumulation first, 2) Then run ExtractStreams
- If a required input doesn't exist, explain what needs to be done first

## LAYER MATCHING
Look at LAYER_CATALOG and copy the EXACT source path. Never invent paths.
"""

# Developer rules for strict behavior
DEVELOPER_RULES = """OUTPUT FORMAT: Return ONLY a JSON object. No markdown. No extra text.

## JSON ACTIONS

1. run_algorithm - Run a processing algorithm
{"action": "run_algorithm", "algorithm_id": "wbt:AlgorithmName", "params": {"param1": "EXACT_PATH_FROM_LAYER_CATALOG", "output": "TEMP"}, "load_outputs": true}

2. ask_user - Ask for missing information
{"action": "ask_user", "question": "What threshold value should I use for stream extraction?"}

3. explain - Answer questions OR explain required steps
{"action": "explain", "text": "Your explanation here"}

## EXAMPLES

LAYER_CATALOG shows:
  - name: dem
    source: /media/hdd/Dropbox/GitHub/whitebox-agents/data/dem.tif

User: "fill depressions"
{"action": "run_algorithm", "algorithm_id": "wbt:FillDepressions", "params": {"dem": "/media/hdd/Dropbox/GitHub/whitebox-agents/data/dem.tif", "output": "TEMP"}, "load_outputs": true}

User: "extract streams from the DEM"
{"action": "explain", "text": "To extract streams, I need to first calculate flow accumulation. Let me do that: run D8FlowAccumulation on your DEM, then I can extract streams with a threshold. Should I proceed with flow accumulation first?"}

User: "what lidar algorithms are available?"
{"action": "explain", "text": "Available LiDAR algorithms include: AsciiToLas, ClassifyBuildingsInLidar, ClassifyLidar, ClipLidarToPolygon, ColourizeBasedOnClass, and more."}

User: "calculate flow accumulation"  
{"action": "run_algorithm", "algorithm_id": "wbt:D8FlowAccumulation", "params": {"input": "/media/hdd/Dropbox/GitHub/whitebox-agents/data/dem.tif", "output": "TEMP"}, "load_outputs": true}

CRITICAL: Copy paths EXACTLY from LAYER_CATALOG. Never use example paths like /media/data/."""


class LLMClient:
    """
    Unified client for multiple LLM providers.

    Handles communication with Ollama, Claude, OpenAI, Gemini, and Amazon Bedrock APIs.
    """

    # Default models for each provider (updated to latest as of Dec 2024)
    DEFAULT_MODELS = {
        LLMProvider.OLLAMA: "llama3.1",
        LLMProvider.CLAUDE: "claude-sonnet-4-20250514",
        LLMProvider.OPENAI: "gpt-4o",
        LLMProvider.GEMINI: "gemini-2.5-flash-preview-05-20",
        LLMProvider.BEDROCK: "anthropic.claude-sonnet-4-20250514-v1:0",
    }

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM client.

        Args:
            config: LLMConfig instance with provider settings.
        """
        self.config = config
        self.model = config.model or self.DEFAULT_MODELS.get(config.provider)
        self._client = None

    def send_message(
        self,
        user_message: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a message to the LLM and get a structured response.

        Args:
            user_message: The user's natural language request.
            context: Context blocks (algorithms, layers, param schema).
            conversation_history: Optional list of previous messages.

        Returns:
            Parsed JSON response from the LLM.
        """
        # Build the full prompt
        full_system = f"{SYSTEM_PROMPT}\n\n{DEVELOPER_RULES}\n\n{context}"

        # Dispatch to the appropriate provider
        if self.config.provider == LLMProvider.OLLAMA:
            response_text = self._send_ollama(
                full_system, user_message, conversation_history
            )
        elif self.config.provider == LLMProvider.CLAUDE:
            response_text = self._send_claude(
                full_system, user_message, conversation_history
            )
        elif self.config.provider == LLMProvider.OPENAI:
            response_text = self._send_openai(
                full_system, user_message, conversation_history
            )
        elif self.config.provider == LLMProvider.GEMINI:
            response_text = self._send_gemini(
                full_system, user_message, conversation_history
            )
        elif self.config.provider == LLMProvider.BEDROCK:
            response_text = self._send_bedrock(
                full_system, user_message, conversation_history
            )
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

        # Parse the response
        return self._parse_response(response_text)

    def _send_ollama(
        self,
        system: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send message to Ollama API."""
        import urllib.request
        import urllib.error

        base_url = self.config.base_url or "http://localhost:11434"

        messages = [{"role": "system", "content": system}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {base_url}: {e}")

    def _send_claude(
        self,
        system: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send message to Claude API."""
        import urllib.request
        import urllib.error

        if not self.config.api_key:
            raise ValueError("Claude API key is required")

        base_url = self.config.base_url or "https://api.anthropic.com"

        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system,
            "messages": messages,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/messages",
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
                return ""
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Claude API: {e}")

    def _send_openai(
        self,
        system: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send message to OpenAI API."""
        import urllib.request
        import urllib.error

        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        base_url = self.config.base_url or "https://api.openai.com"

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                choices = result.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to OpenAI API: {e}")

    def _send_gemini(
        self,
        system: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send message to Gemini API."""
        import urllib.request
        import urllib.error

        if not self.config.api_key:
            raise ValueError("Gemini API key is required")

        base_url = self.config.base_url or "https://generativelanguage.googleapis.com"

        # Build contents for Gemini
        contents = []

        # Add system instruction as first user message
        contents.append(
            {"role": "user", "parts": [{"text": f"System Instructions:\n{system}"}]}
        )
        contents.append(
            {
                "role": "model",
                "parts": [
                    {
                        "text": "I understand. I will follow these instructions and respond only with JSON."
                    }
                ],
            }
        )

        # Add history
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Add current message
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        url = f"{base_url}/v1beta/models/{self.model}:generateContent?key={self.config.api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Gemini API: {e}")

    def _send_bedrock(
        self,
        system: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Send message to Amazon Bedrock API using boto3."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock. Install with: pip install boto3"
            )

        # Create Bedrock runtime client
        # Uses AWS credentials from environment or ~/.aws/credentials
        region = self.config.base_url or "us-east-1"
        client = boto3.client("bedrock-runtime", region_name=region)

        # Build messages for Claude on Bedrock
        messages = []
        if history:
            for msg in history:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}],
                    }
                )

        messages.append(
            {"role": "user", "content": [{"type": "text", "text": user_message}]}
        )

        # Prepare request body for Claude models on Bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        try:
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())
            content = response_body.get("content", [])
            if content:
                return content[0].get("text", "")
            return ""

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Bedrock API: {e}")

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract JSON.

        Args:
            response_text: Raw response text from the LLM.

        Returns:
            Parsed JSON dictionary.
        """
        text = response_text.strip()

        # Try to parse as-is first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try to find JSON object pattern (handles nested objects)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                # Try to fix common issues
                json_str = json_match.group(0)
                # Replace single quotes with double quotes
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # If it looks like a plain text response, wrap it as explain action
        # This handles cases where LLM ignores JSON format instruction
        if text and not text.startswith("{"):
            # Clean up the text - remove any partial JSON attempts
            clean_text = re.sub(r'\{[^}]*$', '', text).strip()
            if clean_text:
                return {
                    "action": "explain",
                    "text": clean_text,
                }

        # Return error response only if we really can't parse anything
        return {
            "action": "explain",
            "text": f"I couldn't understand the response. Please try rephrasing your request.",
            "error": True,
            "raw": text[:500] if text else "Empty response",
        }

    @staticmethod
    def create_config(
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> LLMConfig:
        """
        Create an LLMConfig from string parameters.

        Args:
            provider: Provider name (ollama, claude, openai, gemini).
            api_key: API key for the provider.
            model: Model name to use.
            base_url: Custom base URL.

        Returns:
            LLMConfig instance.
        """
        provider_enum = LLMProvider(provider.lower())
        return LLMConfig(
            provider=provider_enum,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    def test_connection(self) -> bool:
        """
        Test if the LLM connection is working.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            result = self.send_message(
                user_message='Respond with exactly: {"action": "explain", "text": "ok"}',
                context="This is a connection test.",
            )
            return "action" in result
        except Exception:
            return False
