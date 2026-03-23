from typing import Any

import tiktoken
from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.base.llms.types import MessageRole as MetaMessageRole
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI

from boardgame_rules_backend.rag.prompts import DEFAULT_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from boardgame_rules_backend.settings import rag_config


class VllmOpenAICompatibleLLM(OpenAI):
    """
    OpenAI-compatible endpoint (vLLM) with HuggingFace-style model ids.

    LlamaIndex's OpenAI.metadata calls openai_modelname_to_contextsize(), which only
    lists OpenAI model names — custom ids like Qwen/Qwen2.5-7B-Instruct raise ValueError.
    """

    def __init__(self, context_window: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._context_window_override = context_window

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self._context_window_override,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MetaMessageRole.SYSTEM,
        )

    @property
    def _tokenizer(self):  # type: ignore[override]
        """Tiktoken has no mapping for HF ids; use cl100k as a rough fallback."""
        return tiktoken.get_encoding("cl100k_base")


class Generator:
    def __init__(self, temperature: float | None = None):
        cfg = rag_config.llm
        self._api_base = cfg.api_base
        self._model = cfg.model
        self._temperature = temperature if temperature is not None else cfg.temperature
        self._context_window = cfg.context_window

    def _get_llm(self) -> VllmOpenAICompatibleLLM:
        return VllmOpenAICompatibleLLM(
            context_window=self._context_window,
            api_base=self._api_base,
            api_key="EMPTY",
            model=self._model,
            temperature=self._temperature,
        )

    async def generate(self, query: str, context: str) -> str:
        """Generate answer from query and context."""
        llm = self._get_llm()
        prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        response = await llm.achat(messages)
        return response.message.content or "Не удалось получить ответ."
