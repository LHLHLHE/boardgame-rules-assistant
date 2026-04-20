from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from omegaconf import DictConfig, OmegaConf

from src.retriever import Retriever

DEFAULT_SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы о правилах настольных игр. "
    "ВАЖНО: Используй ТОЛЬКО информацию из предоставленного контекста. "
    "НЕ выдумывай детали, которых нет в контексте. "
    "НЕ добавляй информацию из своих знаний, если её нет в контексте. "
    "Если в контексте нет ответа на вопрос, честно скажи: "
    "'В предоставленном контексте нет информации об этом.' "
    "Отвечай кратко, точно и по делу на русском языке, цитируя факты из контекста."
)
NO_CONTEXT_SYSTEM_PROMPT = (
    "Ты помощник. Отвечай кратко на вопросы на русском языке. "
    "Если не знаешь ответ или вопрос не о чём-то общеизвестном - честно скажи, что не знаешь."
)
USER_PROMPT_TEMPLATE = """
Используй ТОЛЬКО информацию из контекста ниже. НЕ выдумывай детали, которых нет в контексте.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ОТВЕТ (используй только факты из контекста выше):"""


def create_eval_llm(cfg: DictConfig, temperature: float | None = None) -> LLM:
    """Создаёт LlamaIndex LLM из eval_llm конфига (генерация датасета / валидация)."""
    default_temp = float(OmegaConf.select(cfg, "eval_llm.temperature", default=0.0))
    temp = temperature if temperature is not None else default_temp
    provider = str(OmegaConf.select(cfg, "eval_llm.provider", default="ollama")).lower()
    if provider != "ollama":
        raise ValueError(
            f"Unknown EVAL_LLM_PROVIDER: {provider}. Supported: 'ollama'"
        )
    base_url = str(OmegaConf.select(cfg, "llm.ollama_base_url", default="http://localhost:11434"))
    model = str(OmegaConf.select(cfg, "eval_llm.model", default="qwen2.5:7b-instruct"))
    max_tokens = int(OmegaConf.select(cfg, "eval_llm.max_tokens", default=512))
    return Ollama(
        model=model,
        base_url=base_url,
        temperature=temp,
        request_timeout=60.0,
        additional_kwargs={"num_predict": max_tokens},
    )


class Generator:
    """Генератор для RAG."""

    def __init__(
        self,
        cfg: DictConfig,
        llm: LLM | None = None,
        retriever: Retriever | None = None,
    ):
        """
        Args:
            cfg: Hydra-конфиг.
            llm: Опционально прединициализированный LLM. Если None - создаётся из конфига.
            retriever: Опциональный retriever. Если None - создаётся новый.
        """
        self.cfg = cfg
        self.llm = llm or self._create_llm()
        self.retriever = retriever or Retriever(cfg)

    def _create_llm(self) -> LLM:
        provider = str(OmegaConf.select(self.cfg, "llm.provider", default="ollama")).lower()
        if provider != "ollama":
            raise ValueError(
                f"Unknown LLM provider: {provider}. Supported: 'ollama'"
            )
        model = str(OmegaConf.select(self.cfg, "llm.model", default="qwen2.5:1.5b"))
        base_url = str(OmegaConf.select(
            self.cfg,
            "llm.ollama_base_url",
            default="http://localhost:11434"
        ))
        temperature = float(OmegaConf.select(self.cfg, "llm.temperature", default=0.0))
        max_tokens = int(OmegaConf.select(self.cfg, "llm.max_tokens", default=512))
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=60.0,
            additional_kwargs={"num_predict": max_tokens},
        )

    def generate(
        self,
        query: str,
        top_k: int | None = None,
        system_prompt: str | None = None,
        game_title: str | None = None,
        game_titles: list[str] | None = None,
    ) -> tuple[str, str]:
        """
        Генерирует ответ с помощью RAG.

        Args:
            query: Вопрос пользователя.
            top_k: Число чанков для извлечения (по умолчанию из config).
            system_prompt: Опциональный кастомный system prompt.
            game_title: Название игры для фильтра по метаданным (один вариант).
            game_titles: Список названий игр для фильтра (приоритет над game_title).

        Returns:
            Кортеж (ответ, контекст, переданный в LLM).
        """
        titles = game_titles if game_titles is not None else ([game_title] if game_title else None)
        context = self.retriever.retrieve_with_context(query, top_k=top_k, game_titles=titles)
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        response = self.llm.chat(messages)
        return (response.message.content or "", context)

    def generate_without_context(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> str:
        """Генерирует ответ только LLM без контекста (ablation: RAG vs vanilla LLM)."""
        if system_prompt is None:
            system_prompt = NO_CONTEXT_SYSTEM_PROMPT
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=query),
        ]
        response = self.llm.chat(messages)
        return response.message.content

    def generate_streaming(
        self,
        query: str,
        top_k: int | None = None,
        system_prompt: str | None = None,
        game_title: str | None = None,
        game_titles: list[str] | None = None,
    ):
        """
        Генерирует ответ стримингом.

        Args:
            query: Вопрос пользователя.
            top_k: Число чанков для извлечения.
            system_prompt: Опциональный system prompt.
            game_title: Название игры для фильтра по метаданным.
            game_titles: Список названий игр для фильтра.

        Yields:
            Токены по мере генерации.
        """
        titles = game_titles if game_titles is not None else ([game_title] if game_title else None)
        context = self.retriever.retrieve_with_context(query, top_k=top_k, game_titles=titles)

        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        response_gen = self.llm.stream_chat(messages)
        for token in response_gen:
            if token.delta:
                yield token.delta
