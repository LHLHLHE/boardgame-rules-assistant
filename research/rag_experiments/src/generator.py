from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from src.config import LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.retriever import Retriever

DEFAULT_SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы о правилах настольных игр. "
    "ВАЖНО: Используй ТОЛЬКО информацию из предоставленного контекста. "
    "НЕ выдумывай детали, которых нет в контексте. "
    "НЕ добавляй информацию из своих знаний, если её нет в контексте. "
    "Если в контексте нет ответа на вопрос, честно скажи: 'В предоставленном контексте нет информации об этом.' "
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


class Generator:
    """
    Simple generator for baseline RAG.
    
    Uses retrieved context to generate answers.
    """

    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None):
        """
        Initialize generator.
        
        Args:
            llm: Optional pre-initialized LLM. If None, creates based on config.
            retriever: Optional retriever instance. If None, creates new one.
        """
        self.llm = llm or self._create_llm()
        self.retriever = retriever or Retriever()

    def _create_llm(self) -> LLM:
        provider = LLM_PROVIDER.lower()
        if provider == "ollama":
            return Ollama(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                request_timeout=60.0,
            )
        elif provider == "openai":
            return OpenAI(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: {LLM_PROVIDER}. "
                "Supported: 'ollama', 'openai'"
            )

    def generate(
        self,
        query: str,
        top_k: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate answer using RAG.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve (default: config.TOP_K)
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated answer
        """
        context = self.retriever.retrieve_with_context(query, top_k=top_k)
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        prompt = USER_PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        
        response = self.llm.chat(messages)
        return response.message.content

    def generate_without_context(
        self,
        query: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate answer using only the LLM, without retrieved context.
        Useful for comparing RAG vs vanilla LLM (ablation).
        """
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
    ):
        """
        Generate answer with streaming (yields tokens).

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            system_prompt: Optional custom system prompt

        Yields:
            Token strings as they are generated
        """
        context = self.retriever.retrieve_with_context(query, top_k=top_k)

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
