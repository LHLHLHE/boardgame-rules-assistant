import json
import logging
import random
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, IO

from langchain_core.documents import Document as LCDocument
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError, model_validator
from tqdm import tqdm

from src.eval_data import chunk_fingerprint, load_qa_dataset_from_jsonl
from src.generator import create_eval_llm

logger = logging.getLogger(__name__)


class SingleHopOutput(BaseModel):
    """Single-hop: skip или Q&A."""

    skip: bool = False
    reason: str | None = None
    question: str | None = None
    ground_truths: list[str] | None = None
    evidence_quote: str | None = None

    @model_validator(mode="after")
    def check_skip_or_qa(self):
        if self.skip:
            if not self.reason:
                raise ValueError("reason required when skip=True")
        else:
            if not (self.question and self.ground_truths and self.evidence_quote):
                raise ValueError(
                    "question, ground_truths, evidence_quote required when skip=False"
                )
        return self


class EvidenceItem(BaseModel):
    """Multi-hop evidence item."""

    fragment: int
    quote: str


class MultiHopOutput(BaseModel):
    """Multi-hop: skip или Q&A."""

    skip: bool = False
    reason: str | None = None
    question: str | None = None
    ground_truths: list[str] | None = None
    evidence: list[EvidenceItem] | None = None


class CriticOutput(BaseModel):
    """Critic validation output."""

    accept: bool
    reasons: list[str] = []


class QADatasetGenerator:
    """Генератор Q&A датасета для оценки RAG: single-hop и multi-hop вопросы через LLM."""

    SINGLE_HOP_SYSTEM = (
        "Ты генерируешь вопросы и эталонные ответы для оценки RAG по правилам настольных игр. "
        "Используй ТОЛЬКО предоставленный фрагмент правил. Никаких внешних знаний. "
        "Сгенерируй ОДИН вопрос, который проверяет правило "
        "(порядок действий, запрет/разрешение, условия, числа/лимиты, последствия). "
        "Запрещены вопросы про стратегию/советы/мнение: слова и конструкции типа "
        "\"лучше\", \"выгоднее\", \"стоит ли\", \"оптимально\", \"тактика\", \"стратегия\". "
        "Вопрос должен быть на русском и содержать точное название игры game_title "
        "(вставь как есть). "
        "Ответ должен быть однозначным и полностью выводиться из фрагмента. "
        "Числовые ответы ОБЯЗАТЕЛЬНО с контекстом: "
        "не \"5\", а \"5 раундов\"; не \"2\", а \"2 очка за плитку\". "
        "Верни строго JSON без текста вокруг, без markdown и пояснений. "
        "Если фрагмент не содержит правил (копирайт, ссылки, контакты, оглавление) "
        "ИЛИ нельзя сделать однозначный вопрос-ответ — "
        "верни {\"skip\": true, \"reason\": \"...\"}. "
        "Иначе верни {\"question\": \"...\", \"ground_truths\": [\"...\"], "
        "\"evidence_quote\": \"...\"} - дословная цитата из фрагмента, 1-3 предложения."
    )

    MULTI_HOP_SYSTEM = (
        "Ты генерируешь ОДИН multi-hop вопрос по правилам настольной игры. "
        "Для ответа ОБЯЗАТЕЛЬНО нужна информация из КАЖДОГО фрагмента. "
        "Используй ТОЛЬКО предоставленные фрагменты, без внешних знаний. "
        "ОБЯЗАТЕЛЬНО включи game_title в вопрос дословно — без этого ответ не будет принят. "
        "Вопрос на русском. Запрещены стратегия/советы/мнение (\"лучше\", \"выгоднее\" и т.п.). "
        "Ответ должен быть однозначным. Числовые ответы ОБЯЗАТЕЛЬНО с контекстом: "
        "не \"5\", а \"5 раундов\". "
        "Верни строго JSON без текста вокруг, без markdown и пояснений. "
        "Если невозможно сделать истинный multi-hop (например, один фрагмент лишний) — "
        "верни {\"skip\": true, \"reason\": \"...\"}. "
        "Иначе верни {\"question\": \"...\", \"ground_truths\": [\"...\"], \"evidence\": [...]}. "
        "ОБЯЗАТЕЛЬНО: в evidence ровно N элементов - столько же, сколько фрагментов. "
        "По одному элементу {\"fragment\": i, \"quote\": \"...\"} на каждый. Не меньше, не больше. "
        "Каждый quote — дословная цитата из соответствующего фрагмента, строго 1-3 предложения. "
        "Quote должен быть точной подстрокой соответствующего фрагмента (не перефразируй).\n\n"
        "Пример (game_title=Каркассон, 2 фрагмента):\n"
        "{\"question\": \"Сколько очков приносит завершённый город в игре Каркассон "
        "и когда их получают?\", "
        "\"ground_truths\": [\"2 очка за каждую плитку города, в конце партии\"], "
        "\"evidence\": [{\"fragment\": 1, \"quote\": \"Завершённый город приносит "
        "2 очка за каждую плитку.\"}, "
        "{\"fragment\": 2, \"quote\": \"Очки за город начисляют в конце партии.\"}]}\n\n"
        "Пример для 3 фрагментов (структура та же — ровно 3 элемента в evidence):\n"
        "{\"question\": \"В игре Сеттер Сколько карт раздают в начале партии, "
        "какой ход считается первым и когда игрок может взять карту из колоды?\", "
        "\"ground_truths\": [\"по 6 карт\", \"первый ход за сдатчиком\", \"в свой ход\"], "
        "\"evidence\": [{\"fragment\": 1, \"quote\": \"Каждому игроку сдают по 6 карт.\"}, "
        "{\"fragment\": 2, \"quote\": \"Первый ход делает игрок слева от сдатчика.\"}, "
        "{\"fragment\": 3, \"quote\": \"В свой ход можно взять верхнюю карту из колоды.\"}]}"
    )

    SINGLE_HOP_USER_TEMPLATE = """game_title: {game_title}

    Фрагмент правил:
    {context}
    """

    MULTI_HOP_USER_TEMPLATE = """game_title: {game_title}
    Вопрос ОБЯЗАТЕЛЬНО должен содержать это название игры дословно.

    Фрагментов: {n_fragments}.
    В evidence верни ровно {n_fragments} элементов — по одному на каждый фрагмент.

    Фрагменты правил (1..{n_fragments}). Каждый фрагмент отделяй строкой '---':
    {context}
    """

    CRITIC_SYSTEM = (
        "Ты валидатор синтетического Q&A для оценки RAG по правилам настольных игр. "
        "Тебе дан game_title, контекст(ы) и кандидат (question, ground_truths, evidence...). "
        "Проверь: (1) нет стратегии/мнения; (2) ответы однозначны и числа с контекстом "
        "(не просто \"5\", а \"5 раундов\"); "
        "(3) evidence-цитаты в соответствующих фрагментах "
        "(дословно или с допустимыми отличиями в пунктуации/пробелах); "
        "(4) ответы следуют из evidence/контекста без внешних знаний; "
        "(5) для multi-hop обязательно использование всех фрагментов: "
        "отклони, если любой фрагмент избыточен или не нужен для ответа. "
        "Верни строго JSON без текста вокруг: {\"accept\": true/false, \"reasons\": [\"...\"]}."
    )
    CRITIC_USER_TEMPLATE = """game_title: {game_title}

    Контекст(ы):
    {context}

    Кандидат JSON:
    {candidate_json}
    """

    FORBIDDEN_WORDS_RE = re.compile(
        r"\b(лучше|выгодн\w*|стоит\s+ли|оптимальн\w*|тактик\w*|стратег\w*)\b",
        re.IGNORECASE,
    )

    def _get_game_title(self, doc: LCDocument) -> str | None:
        """Извлекает название игры из metadata."""
        titles = (doc.metadata or {}).get("game_titles") or []
        if isinstance(titles, str):
            titles = [titles]
        titles = [str(t).strip() for t in titles if t and str(t).strip()]
        if titles:
            return titles[0]
        return None

    def __init__(
        self,
        cfg: DictConfig,
        llm: LLM | None = None,
        single_hop_ratio: float = 0.7,
        max_retries: int = 1,
        critic_llm: LLM | None = None,
    ):
        """
        Args:
            cfg: Hydra config
            llm: LLM для генерации. Если None - создаётся через create_eval_llm(cfg).
            single_hop_ratio: Доля single-hop вопросов (0..1).
            max_retries: Число повторов при ошибке парсинга JSON на один сэмпл.
            critic_llm: LLM для filter_with_critic (temperature=0).
                Если None - create_eval_llm(cfg, temperature=0).
        """
        self.llm = llm or create_eval_llm(cfg)
        self.single_hop_ratio = single_hop_ratio
        self.max_retries = max_retries
        self.critic_llm = critic_llm or create_eval_llm(cfg, temperature=0)
        self.multi_chunks_min = int(OmegaConf.select(cfg, "eval.multi_chunks_min", default=2))
        self.multi_chunks_max = int(OmegaConf.select(cfg, "eval.multi_chunks_max", default=2))

    @staticmethod
    def _extract_json_from_response(response: str) -> dict | None:
        text = (response or "").strip()
        # Убирает markdown-блок кода (```json ... ``` или ``` ... ```)
        for pattern in (r"```(?:json)?\s*\n?(.*?)\n?```", r"```(.*?)```"):
            m = re.search(pattern, text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break
        start = text.find("{")
        end = text.rfind("}") + 1
        if end > start >= 0:
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                # Убрать trailing comma перед } или ]
                chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_generation(data: dict, is_multi_hop: bool = False) -> dict | None:
        """Парсит ответ LLM: skip или валидный Q&A с evidence."""
        if not isinstance(data, dict):
            return None
        if data.get("skip") is True:
            return {"skip": True, "reason": str(data.get("reason", "")).strip()}

        q = data.get("question")
        if not isinstance(q, str) or not q.strip():
            return None
        question = q.strip()

        gts = data.get("ground_truths")
        if gts is None:
            gts = data.get("reference") or data.get("reference_answer")
        if isinstance(gts, str):
            gts = [gts]
        if not (isinstance(gts, list) and 1 <= len(gts) <= 2):
            return None
        ground_truths = [str(x).strip() for x in gts if str(x).strip()]
        if not ground_truths:
            return None

        evq = data.get("evidence_quote")
        evidence = data.get("evidence")

        out: dict[str, Any] = {
            "skip": False,
            "question": question,
            "ground_truths": ground_truths,
            "evidence_quote": evq if isinstance(evq, str) else None,
            "evidence": evidence if isinstance(evidence, list) else None,
        }
        return out

    @staticmethod
    def _pydantic_to_parsed(
        obj: SingleHopOutput | MultiHopOutput,
        is_multi_hop: bool
    ) -> dict[str, Any]:
        """Преобразует Pydantic-модель в формат {skip, question, ground_truths, ...}."""
        if obj.skip:
            return {"skip": True, "reason": (obj.reason or "").strip()}
        if is_multi_hop and isinstance(obj, MultiHopOutput):
            evidence = obj.evidence
            evidence_list = (
                [{"fragment": e.fragment, "quote": e.quote} for e in (evidence or [])]
            )
            return {
                "skip": False,
                "question": (obj.question or "").strip(),
                "ground_truths": obj.ground_truths or [],
                "evidence_quote": None,
                "evidence": list(evidence_list),
            }
        return {
            "skip": False,
            "question": (obj.question or "").strip(),
            "ground_truths": obj.ground_truths or [],
            "evidence_quote": (getattr(obj, "evidence_quote", None) or "").strip() or None,
            "evidence": None,
        }

    def _generate_single_hop(self, doc: LCDocument) -> dict | None:
        """Генерирует один Q&A сэмпл по одному чанку. None при ошибке."""
        game_title = self._get_game_title(doc)
        if not game_title:
            return None
        context = (doc.page_content or "").strip()
        prompt = self.SINGLE_HOP_USER_TEMPLATE.format(game_title=game_title, context=context)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.SINGLE_HOP_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        response = self.llm.chat(messages, format="json")
        raw = (response.message.content or "").strip()
        try:
            obj = SingleHopOutput.model_validate_json(raw)
            parsed = self._pydantic_to_parsed(obj, is_multi_hop=False)
        except ValidationError:
            data = self._extract_json_from_response(raw)
            parsed = self._parse_generation(data, is_multi_hop=False)
        if not parsed:
            logger.debug("Single-hop parse failed. Raw (first 500 chars): %s", raw[:500])
            return None
        return parsed

    def _generate_multi_hop(
        self,
        docs: list[LCDocument],
        hint: str | None = None,
    ) -> dict | None:
        """Генерирует один Q&A сэмпл по 2–3 чанкам. None при ошибке."""
        game_title = self._get_game_title(docs[0])
        if not game_title:
            return None
        parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content or ""
            parts.append(f"--- Фрагмент {i} ---\n{content}")
        context = "\n\n".join(parts)
        n_fragments = len(docs)
        prompt = self.MULTI_HOP_USER_TEMPLATE.format(
            game_title=game_title,
            context=context,
            n_fragments=n_fragments
        )
        if hint:
            prompt = f"ВАЖНО: {hint}\n\n{prompt}"
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.MULTI_HOP_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        response = self.llm.chat(messages, format="json")
        raw = (response.message.content or "").strip()
        try:
            obj = MultiHopOutput.model_validate_json(raw)
            parsed = self._pydantic_to_parsed(obj, is_multi_hop=True)
        except ValidationError:
            data = self._extract_json_from_response(raw)
            parsed = self._parse_generation(data, is_multi_hop=True)
        if not parsed:
            logger.debug("Multi-hop parse failed. Raw (first 500 chars): %s", raw[:500])
            return None
        return parsed

    @staticmethod
    def _normalize_for_quote_match(s: str) -> str:
        """
        Нормализация для проверки цитаты.

        Цель: снизить ложные несовпадения из-за регистра/юникода/пунктуации/пробелов,
        но при этом оставаться строгими к перефразированию.
        """
        norm = unicodedata.normalize("NFKC", (s or ""))
        norm = norm.strip().lower().replace("ё", "е")
        norm = re.sub(r"\s+", " ", norm)
        return re.sub(r"[^\w\s]", "", norm)

    BARE_NUMERIC_RE = re.compile(r"^\s*[\d\s\-,\.]+\s*$")

    @staticmethod
    def _is_bare_numeric(s: str) -> bool:
        """True, если строка - только число без контекста (например \"5\", \"3-5\")."""
        if not s or not s.strip():
            return False
        return bool(QADatasetGenerator.BARE_NUMERIC_RE.match(s))

    @staticmethod
    def _game_title_in_question(question: str, game_title: str) -> bool:
        """
        Проверяет, что game_title присутствует в вопросе.
        - Точное совпадение или полное название как подстрока.
        - Для составных (через '. '): каждая часть ИЛИ хотя бы первая (основное название).
        """
        if not game_title or not question:
            return False
        if game_title in question:
            return True
        q_lower = question.lower()
        parts = [p.strip() for p in game_title.split(". ") if p.strip()]
        if len(parts) >= 2:
            if all(p.lower() in q_lower for p in parts):
                return True
            # Частичное: достаточно первого сегмента (основное название)
            if parts[0].lower() in q_lower and len(parts[0]) >= 3:
                return True
        return game_title.lower() in q_lower

    @staticmethod
    def _is_mostly_russian(text: str, min_cyrillic_ratio: float = 0.3) -> bool:
        """
        Детерминированная проверка, что текст в основном на русском.

        Считаем долю кириллицы среди буквенных символов (без цифр/пробелов/пунктуации),
        чтобы наличие латиницы в названии игры или артефактов не ломало проверку.
        """
        if not text:
            return False
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04FF")
        return (cyr / len(letters)) >= float(min_cyrillic_ratio)

    def _fast_validate_single_hop(
        self,
        game_title: str,
        context: str,
        out: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Быстрые детерминированные проверки single-hop. Возвращает (ok, reason)."""
        question = (out.get("question") or "").strip()
        if not self._game_title_in_question(question, game_title):
            out["question"] = f"В игре {game_title} {question}".strip()
            question = out["question"]
        if not self._is_mostly_russian(question):
            return False, "non-russian question"
        if self.FORBIDDEN_WORDS_RE.search(question):
            return False, "forbidden words (strategy/advice)"
        gts = out.get("ground_truths") or []
        for i, gt in enumerate(gts):
            if isinstance(gt, str) and self._is_bare_numeric(gt):
                return False, f"ground_truths[{i}] bare numeric (need context)"
        evq = (out.get("evidence_quote") or "").strip()
        if not evq:
            return False, "evidence_quote empty"
        ctx_norm = self._normalize_for_quote_match(context)
        evq_norm = self._normalize_for_quote_match(evq)
        if evq_norm not in ctx_norm:
            return False, "evidence_quote not substring of context"
        return True, None

    def _fast_validate_multi_hop(
        self,
        game_title: str,
        fragments: list[str],
        out: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Быстрые детерминированные проверки multi-hop. Возвращает (ok, reason)."""
        question = (out.get("question") or "").strip()
        if not self._game_title_in_question(question, game_title):
            out["question"] = f"В игре {game_title} {question}".strip()
            question = out["question"]
        if not self._is_mostly_russian(question):
            return False, "non-russian question"
        if self.FORBIDDEN_WORDS_RE.search(question):
            return False, "forbidden words (strategy/advice)"
        gts = out.get("ground_truths") or []
        for i, gt in enumerate(gts):
            if isinstance(gt, str) and self._is_bare_numeric(gt):
                return False, f"ground_truths[{i}] bare numeric (need context)"
        evidence = out.get("evidence")
        n_frag = len(fragments)
        n_ev = len(evidence) if isinstance(evidence, list) else 0
        if not isinstance(evidence, list) or n_ev != n_frag:
            reason = f"evidence length mismatch (fragments={n_frag}, evidence={n_ev})"
            return False, reason
        for i, item in enumerate(evidence):
            if not isinstance(item, dict):
                return False, f"evidence[{i}] not dict"
            quote = (item.get("quote") or "").strip()
            if not quote:
                return False, f"evidence[{i}] quote empty"
            frag_idx = item.get("fragment")
            if frag_idx is not None:
                frag_idx = int(frag_idx) - 1
            else:
                frag_idx = i
            if frag_idx < 0 or frag_idx >= len(fragments):
                return False, f"evidence[{i}] bad fragment index"
            frag_norm = self._normalize_for_quote_match(fragments[frag_idx])
            quote_norm = self._normalize_for_quote_match(quote)
            if quote_norm not in frag_norm:
                return False, f"evidence[{i}] quote not in fragment"
        return True, None

    def _critic_validate(
        self,
        game_title: str,
        context: str,
        candidate: dict[str, Any],
        is_multi_hop: bool,
    ) -> tuple[bool, list[str]]:
        """LLM-критик: проверяет качество кандидата. Возвращает (accept, reasons)."""
        candidate_json = json.dumps(candidate, ensure_ascii=False)
        prompt = self.CRITIC_USER_TEMPLATE.format(
            game_title=game_title,
            context=context,
            candidate_json=candidate_json,
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.CRITIC_SYSTEM),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        response = self.critic_llm.chat(messages, format="json")
        raw = (response.message.content or "").strip()
        try:
            obj = CriticOutput.model_validate_json(raw)
            if obj.accept:
                return True, obj.reasons or []
            return False, obj.reasons or ["critic reject"]
        except ValidationError:
            data = self._extract_json_from_response(raw)
            if not isinstance(data, dict):
                return False, ["critic JSON parse failed"]
            accept = data.get("accept")
            if accept is True:
                return True, data.get("reasons") or []
            reasons = data.get("reasons")
            if isinstance(reasons, list):
                reasons = [str(r) for r in reasons]
            else:
                reasons = [str(reasons)] if reasons else ["critic reject"]
            return False, reasons

    @staticmethod
    def _metadata_for_gold_context(metadata: dict) -> dict[str, Any]:
        """Сериализует метаданные для gold_contexts."""
        out: dict[str, Any] = {}
        for key in ("game_titles", "lang", "source_doc_id"):
            value = metadata.get(key)
            if value is None:
                continue
            if key == "game_titles" and isinstance(value, str):
                out[key] = [value] if value.strip() else []
            elif isinstance(value, (str, int, float, bool)) or (
                isinstance(value, list)
                and all(isinstance(x, (str, int, float, bool)) for x in value)
            ):
                out[key] = value
        return out

    def _lcdoc_to_gold_context(self, doc: LCDocument) -> dict[str, Any]:
        """Преобразовать LCDocument в элемент gold_contexts (text, fingerprint, metadata)."""
        raw_content = doc.page_content or ""
        text = raw_content.strip()
        return {
            "text": text,
            "fingerprint": chunk_fingerprint(text),
            "metadata": self._metadata_for_gold_context(doc.metadata or {}),
        }

    def _chunks_by_doc_id(self, chunks: list[LCDocument]) -> dict[str, list[LCDocument]]:
        """Группирует чанки по source_doc_id для multi-hop. Пропускает чанки без source_doc_id."""
        by_id: dict[str, list[LCDocument]] = {}
        for doc in chunks:
            source_doc_id = (doc.metadata or {}).get("source_doc_id")
            if not source_doc_id:
                continue
            if source_doc_id not in by_id:
                by_id[source_doc_id] = []
            by_id[source_doc_id].append(doc)
        result: dict[str, list[LCDocument]] = {}
        for doc_id, doc_chunks in by_id.items():
            if len(doc_chunks) < 2:
                continue
            titles = {self._get_game_title(d) for d in doc_chunks}
            if None in titles or len(titles) > 1:
                continue
            result[doc_id] = doc_chunks
        return result

    def generate(
        self,
        chunks: list[LCDocument],
        testset_size: int,
        random_seed: int | None = None,
        out_file: IO[str] | None = None,
        flush_every: int = 10,
        start_idx: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Генерирует Q&A датасет (single-hop + multi-hop).

        Args:
            chunks: Список LCDocument.
            testset_size: Целевое число сэмплов.
            random_seed: Seed для воспроизводимости.
            out_file: Если задан, каждая запись пишется в файл по мере появления.
            flush_every: После скольких записей вызывать out_file.flush() (при заданном out_file).
            start_idx: С какого номера начинать id (для возобновления).

        Returns:
            Список dict с полями id, question, ground_truths, gold_contexts.
        """
        rng = random.Random(random_seed)

        n_single = max(0, int(round(testset_size * self.single_hop_ratio)))
        n_multi = max(0, testset_size - n_single)

        by_doc_id = self._chunks_by_doc_id(chunks)
        doc_ids_with_multi = list(by_doc_id.keys())
        rng.shuffle(doc_ids_with_multi)

        records: list[dict[str, Any]] = []
        idx = 0
        skipped = 0
        rejected = 0
        with tqdm(total=testset_size, desc="Генерация сэмплов", unit="samp") as pbar:
            single_done = 0
            while single_done < n_single:
                if not chunks:
                    break
                doc = rng.choice(chunks)
                for attempt in range(self.max_retries + 1):
                    out = self._generate_single_hop(doc)
                    if out and out.get("skip"):
                        skipped += 1
                        break
                    if out:
                        context = (doc.page_content or "").strip()
                        game_title = self._get_game_title(doc)
                        if game_title:
                            ok, reason = self._fast_validate_single_hop(
                                game_title, context, out
                            )
                            if not ok:
                                rejected += 1
                                if attempt < self.max_retries:
                                    q = (out.get("question") or "")
                                    logger.debug(
                                        "Single-hop reject: %s | game_title=%r | question=%s",
                                        reason,
                                        game_title,
                                        q,
                                    )
                                    continue
                                break
                        gold_contexts = [self._lcdoc_to_gold_context(doc)]
                        record = {
                            "id": f"{start_idx + idx:06d}",
                            "question_type": "single_hop",
                            "game_title": (game_title or ""),
                            "question": out["question"],
                            "ground_truths": out["ground_truths"],
                            "gold_contexts": gold_contexts,
                        }
                        if out.get("evidence_quote"):
                            record["evidence_quote"] = out["evidence_quote"]
                        records.append(record)
                        if out_file is not None:
                            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            if len(records) % flush_every == 0:
                                out_file.flush()
                        idx += 1
                        single_done += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            ok=len(records),
                            skipped=skipped,
                            rejected=rejected,
                        )
                        break
                    if attempt < self.max_retries:
                        logger.debug("Single-hop JSON parse failed, retrying once.")
                else:
                    logger.debug("Single-hop sample skipped after retries.")

            multi_chunks_min, multi_chunks_max = self.multi_chunks_min, self.multi_chunks_max
            multi_done = 0
            while multi_done < n_multi:
                if not doc_ids_with_multi:
                    break
                doc_id = rng.choice(doc_ids_with_multi)
                group = by_doc_id[doc_id]
                n_use = min(rng.randint(multi_chunks_min, multi_chunks_max), len(group))
                docs = rng.sample(group, n_use)
                hint: str | None = None
                for attempt in range(self.max_retries + 1):
                    out = self._generate_multi_hop(docs, hint=hint)
                    hint = None
                    if out and out.get("skip"):
                        skipped += 1
                        break
                    if out:
                        game_title = self._get_game_title(docs[0])
                        fragments = [
                            (d.page_content or "").strip()
                            for d in docs
                        ]
                        if game_title:
                            ok, reason = self._fast_validate_multi_hop(
                                game_title, fragments, out
                            )
                            if not ok:
                                rejected += 1
                                if attempt < self.max_retries:
                                    q = (out.get("question") or "")
                                    ev = out.get("evidence")
                                    ev_preview = ""
                                    if "evidence length" in (reason or ""):
                                        n_frag = len(fragments)
                                        n_ev = len(ev) if isinstance(ev, list) else 0
                                        hint = (
                                            f"Ты вернул evidence длины {n_ev} при N={n_frag}. "
                                            f"Верни ровно {n_frag} элемента evidence, "
                                            f"с fragment=1..{n_frag}, quote непустые, дословные."
                                        )
                                        ev_preview = " | evidence_frag_ids="
                                        if isinstance(ev, list) and ev:
                                            ids = [str(e.get("fragment", "?")) for e in ev[:8]]
                                            ev_preview += str(ids)
                                        else:
                                            ev_preview += "not_list"
                                    logger.debug(
                                        "Multi-hop reject: %s | game_title=%r | question=%s%s",
                                        reason,
                                        game_title,
                                        q,
                                        ev_preview,
                                    )
                                    continue
                                break
                        gold_contexts = [self._lcdoc_to_gold_context(d) for d in docs]
                        record = {
                            "id": f"{start_idx + idx:06d}",
                            "question_type": "multi_hop",
                            "game_title": (game_title or ""),
                            "question": out["question"],
                            "ground_truths": out["ground_truths"],
                            "gold_contexts": gold_contexts,
                        }
                        if out.get("evidence"):
                            record["evidence"] = out["evidence"]
                        records.append(record)
                        if out_file is not None:
                            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                            if len(records) % flush_every == 0:
                                out_file.flush()
                        idx += 1
                        multi_done += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            ok=len(records),
                            skipped=skipped,
                            rejected=rejected,
                        )
                        break
                    if attempt < self.max_retries:
                        logger.debug("Multi-hop JSON parse failed, retrying once.")
                else:
                    logger.debug("Multi-hop sample skipped after retries.")

        return records

    def filter_with_critic(
        self,
        records: list[dict[str, Any]],
        out_file: IO[str] | None = None,
        flush_every: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Прогон критика по уже сгенерированным записям. Возвращает только принятые.
        """
        accepted: list[dict[str, Any]] = []
        reject_reasons: Counter[str] = Counter()
        for rec in tqdm(records, desc="Critic pass", unit="samp"):
            gold = rec.get("gold_contexts") or []
            if not gold:
                logger.debug("Record %s has no gold_contexts, skip critic", rec.get("id"))
                continue
            gt_meta = gold[0].get("metadata") or {}
            titles = gt_meta.get("game_titles") or []
            game_title = (titles[0] if isinstance(titles, list) else titles) if titles else ""
            if isinstance(game_title, list):
                game_title = game_title[0] if game_title else ""
            game_title = str(game_title).strip() if game_title else ""
            if not game_title:
                logger.debug("Record %s has no game_title in metadata, skip critic", rec.get("id"))
                accepted.append(rec)
                continue
            is_multi = rec.get("question_type") == "multi_hop"
            if is_multi:
                context = "\n\n".join(
                    f"--- Фрагмент {j} ---\n{(g.get('text') or '').strip()}"
                    for j, g in enumerate(gold, 1)
                )
            else:
                context = (gold[0].get("text") or "").strip()
            candidate = {
                "question": rec.get("question", ""),
                "ground_truths": rec.get("ground_truths", []),
                "evidence_quote": rec.get("evidence_quote"),
                "evidence": rec.get("evidence"),
            }
            accept, reasons = self._critic_validate(
                game_title, context, candidate, is_multi_hop=is_multi
            )
            if accept:
                accepted.append(rec)
                if out_file is not None:
                    out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if len(accepted) % flush_every == 0:
                        out_file.flush()
            else:
                reasons = reasons or ["critic reject"]
                reject_reasons.update(reasons)
                logger.debug(
                    "Critic reject: id=%s | type=%s | game_title=%r | reasons=%s | question=%s",
                    rec.get("id"),
                    rec.get("question_type"),
                    game_title,
                    reasons,
                    (rec.get("question") or ""),
                )
        if reject_reasons:
            top = ", ".join(f"{k}={v}" for k, v in reject_reasons.most_common(10))
            logger.info("Critic reject reasons (top): %s", top)
        return accepted


def run_critic_pass(
    cfg: DictConfig,
    input_path: str | Path,
    output_path: str | Path | None = None,
    critic_llm: LLM | None = None,
    flush_every: int = 10,
    only_new_count: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Прогон критика по JSONL-датасету. Фильтрует и сохраняет только принятые записи.

    Если only_new_count задан - критик проверяет только последние only_new_count записей,
    остальные сохраняются без проверки.
    """
    input_path = Path(input_path)
    records = load_qa_dataset_from_jsonl(input_path)
    gen = QADatasetGenerator(cfg, critic_llm=critic_llm or create_eval_llm(cfg, temperature=0))
    out_path = Path(output_path) if output_path else input_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".critic_tmp")

    if only_new_count is not None and only_new_count > 0:
        n = min(only_new_count, len(records))
        old_records = records[:-n] if n < len(records) else []
        to_check = records[-n:]
        accepted_new = gen.filter_with_critic(to_check, out_file=None, flush_every=flush_every)
        accepted = accepted_new
        total_checked = len(to_check)
        to_write = old_records + accepted_new
    else:
        total_checked = len(records)
        to_write = None

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if to_write is not None:
                for rec in to_write:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                accepted = gen.filter_with_critic(records, out_file=f, flush_every=flush_every)
        shutil.move(str(tmp_path), str(out_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return accepted, total_checked


def generate_qa_dataset(
    cfg: DictConfig,
    chunks: list[LCDocument],
    testset_size: int,
    single_hop_ratio: float = 0.7,
    llm: LLM | None = None,
    random_seed: int | None = None,
    max_retries: int = 1,
    out_file: IO[str] | None = None,
    flush_every: int = 10,
    start_idx: int = 0,
    critic_llm: LLM | None = None,
) -> list[dict[str, Any]]:
    """
    Генерирует Q&A датасет.

    Returns:
        Список dict с id, question, ground_truths, gold_contexts, [evidence_quote], [evidence].
    """
    gen = QADatasetGenerator(
        cfg=cfg,
        llm=llm,
        single_hop_ratio=single_hop_ratio,
        max_retries=max_retries,
        critic_llm=critic_llm,
    )
    return gen.generate(
        chunks,
        testset_size,
        random_seed=random_seed,
        out_file=out_file,
        flush_every=flush_every,
        start_idx=start_idx,
    )
