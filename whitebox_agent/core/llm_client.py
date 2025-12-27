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
SYSTEM_PROMPT = """You are an AI assistant that executes WhiteboxTools algorithms through QGIS.

## RESPONSE FORMAT
ALWAYS respond with exactly ONE valid JSON object. No extra text before or after.

## RULES
1. If input layer exists in LAYER_CATALOG → run algorithm immediately (don't ask!)
2. Use EXACT paths from LAYER_CATALOG (the "source:" value)
3. Use EXACT algorithm_id from AVAILABLE_ALGORITHMS 
4. When user confirms ("yes", "ok", "please do") → run algorithm immediately
5. For continuation messages → use the provided output path as input

## COMMON PARAMETERS
- wbt:FillDepressions: dem (not input!)
- wbt:D8FlowAccumulation: input
- wbt:ExtractStreams: flow_accum (not input!), threshold
- wbt:WetnessIndex: sca (=flow_accum), slope (use DEM path if no slope layer exists!)
- wbt:Hillshade: dem
- Always use "output": "TEMP"
- NEVER invent paths like "slope.tif" - only use paths that exist in LAYER_CATALOG!

## MULTI-STEP WORKFLOWS  
- If flow_accum layer exists in LAYER_CATALOG, use it directly for ExtractStreams/WetnessIndex
- If not, run D8FlowAccumulation first, then continue with the original request
"""

# Developer rules for strict behavior
DEVELOPER_RULES = """## RESPONSE FORMAT
Return ONLY ONE JSON object. No markdown, no extra text, no multiple JSON objects.

## ACTIONS
1. run_algorithm: {"action": "run_algorithm", "algorithm_id": "wbt:AlgName", "params": {"param": "VALUE", "output": "TEMP"}, "load_outputs": true}
2. explain: {"action": "explain", "text": "Your message"}
3. ask_user: {"action": "ask_user", "question": "Your question"} (use sparingly!)

## WHEN TO USE EACH
- Input exists in LAYER_CATALOG → run_algorithm immediately
- User confirms ("yes", "ok", "please do") → run_algorithm immediately  
- User asks a question → explain
- Need clarification on which layer → ask_user

## EXAMPLES (paths below are FAKE - always use REAL paths from LAYER_CATALOG!)

"fill depressions": 
{"action": "run_algorithm", "algorithm_id": "wbt:FillDepressions", "params": {"dem": "<DEM_PATH>", "output": "TEMP"}, "load_outputs": true}

"flow accumulation":
{"action": "run_algorithm", "algorithm_id": "wbt:D8FlowAccumulation", "params": {"input": "<DEM_PATH>", "output": "TEMP"}, "load_outputs": true}

"extract streams" (when flow accum layer exists):
{"action": "run_algorithm", "algorithm_id": "wbt:ExtractStreams", "params": {"flow_accum": "<FLOW_ACCUM_PATH>", "threshold": 1000, "output": "TEMP"}, "load_outputs": true}

"extract streams" (no flow accum layer):
{"action": "explain", "text": "ExtractStreams requires flow accumulation. Should I run D8FlowAccumulation first?"}

"calculate wetness" (when flow accum exists):
→ WetnessIndex needs: sca=flow_accum, slope=DEM (use DEM for slope if no slope layer!)
{"action": "run_algorithm", "algorithm_id": "wbt:WetnessIndex", "params": {"sca": "<FLOW_ACCUM_PATH>", "slope": "<DEM_PATH>", "output": "TEMP"}, "load_outputs": true}

"calculate wetness" (no flow accum):
{"action": "explain", "text": "WetnessIndex requires flow accumulation. Should I run D8FlowAccumulation first?"}

## CRITICAL
- NEVER use "<PATH_FROM_LAYER_CATALOG>" literally - replace it with the ACTUAL path from LAYER_CATALOG!
- Find the layer in LAYER_CATALOG and copy its exact "source:" value
- Use "output": "TEMP" always
- Check AVAILABLE_ALGORITHMS for exact param names (flow_accum not input, sca not dem)
- Algorithm IDs are case-sensitive (wbt:Hillshade not wbt:HillShade)"""


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

        # Try to find the FIRST complete JSON object (handles multiple JSON objects in response)
        # Use a more careful approach to find balanced braces
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    json_str = text[start_idx:i+1]
                    try:
                        result = json.loads(json_str)
                        # Only return if it has a valid action
                        if "action" in result:
                            return result
                    except json.JSONDecodeError:
                        pass
                    # Reset and keep looking
                    start_idx = -1

        # Fallback: try simple regex for JSON object
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if "action" in result:
                    return result
            except json.JSONDecodeError:
                pass

        # If it looks like a plain text response, wrap it as explain action
        if text and not text.startswith("{"):
            # Clean up the text - remove any JSON attempts
            clean_text = re.sub(r"\{.*?\}", "", text, flags=re.DOTALL).strip()
            clean_text = clean_text[:500] if clean_text else text[:500]
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
