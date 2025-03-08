import asyncio
import aiohttp
import json
from typing import Any, Protocol, Sequence, TypedDict, Literal
from dataclasses import dataclass

class Message(TypedDict):
    role: Literal["user"] | Literal["assistant"]
    content: str

class ChatClient(Protocol):
    async def chat(
        self,
        prompt: str
    ) -> Sequence[Message]: ...

@dataclass
class OpenAiCompatClient(ChatClient):
    """An OpenAI-compatible API client"""
    api_key: str
    base_url: str
    model: str
    temperature: float = 1.0
    retries: int = 3

    async def chat(
        self,
        prompt: str
    ) -> Sequence[Message]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        content = ""

        async for line in make_request(
            url=url,
            headers=headers,
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": self.temperature,
            },
            retries=self.retries,
        ):
            json_data = json.loads(line)
            if (
                "choices" in json_data
                and json_data["choices"]
                and "delta" in json_data["choices"][0]
                and "content" in json_data["choices"][0]["delta"]
            ):
                content += json_data["choices"][0]["delta"]["content"]
            elif "error" in json_data:
                raise Exception(json_data["error"])

        return [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": content
            },
        ]

@dataclass
class AnthropicCompatClient(ChatClient):
    """An Anthropic-compatible API client. If thinking_budget is set, streams
    down thinking tokens and outputs them inside <think> tags in the assistant
    response object"""
    api_key: str
    model: str
    max_tokens: int
    thinking_budget: None | int = None
    base_url: str = "https://api.anthropic.com/v1"
    temperature: float = 1.0
    retries: int = 3

    async def chat(
        self,
        prompt: str
    ) -> Sequence[Message]:
        url = f"{self.base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        content = ""
        thoughts = None
        async for line in make_request(
            url=url,
            headers=headers,
            payload={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "thinking": None if self.thinking_budget is None else {
                    "type": "enabled",
                    "budget_tokens": self.thinking_budget,
                }
            },
            retries=self.retries,
        ):
            json_data = json.loads(line)
            if "type" in json_data:
                if json_data["type"] == "error":
                    raise Exception(json_data["error"])
                elif json_data["type"] == "content_block_delta":
                    if json_data["delta"]["type"] == "text_delta":
                        content += json_data["delta"]["text"]
                    if json_data["delta"]["type"] == "thinking_delta":
                        if thoughts is None:
                            thoughts = json_data["delta"]["thinking"]
                        else:
                            thoughts = thoughts + json_data["delta"]["thinking"]

        return [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": f"<think>{thoughts}</think>\n{content}" if thoughts else content
            },
        ]

async def make_request(
    url: str,
    headers: dict[str, str],
    payload: Any,
    retries: int,
):
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    async for linebytes in response.content:
                        line = linebytes.decode('utf-8').strip()
                        if line:
                            # SSE format starts with "data: "
                            if line.startswith('data: '):
                                if line == 'data: [DONE]':
                                    break
                                yield line[6:]
                            elif line.startswith('{'):
                                yield line
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                raise
            await asyncio.sleep(1 * (attempt + 1))
