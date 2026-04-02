import html

from aiogram import F, Router
from aiogram.enums import MessageEntityType
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import (BufferedInputFile, CallbackQuery, InlineQuery, InlineQueryResultArticle,
                           InputFile, InputTextMessageContent, Message)

from boardgame_rules_bot.backend import ask_question, download_rules_source, fetch_games
from boardgame_rules_bot.config import settings
from boardgame_rules_bot.constants import (CALLBACK_ACTION_ASK, CALLBACK_ACTION_CANCEL,
                                           CALLBACK_ACTION_DOWNLOAD_SOURCE, HELP_TEXT,
                                           INLINE_GAME_URL_PREFIX)
from boardgame_rules_bot.keyboards import (build_inline_search_keyboard, build_start_keyboard,
                                           build_waiting_question_keyboard)
from boardgame_rules_bot.states import AskStates
from boardgame_rules_bot.utils import build_history_text, clip_text

router = Router()


ASK_PROMPT_TEXT = (
    "Нажмите кнопку «Найти игру» и начните вводить название игры.\n"
    "Telegram покажет результаты прямо над полем ввода."
)

NO_GAME_CONTEXT_TEXT = (
    "Сейчас не выбрана игра: ответы по правилам не ищутся.\n\n"
    "Нажмите «Задать вопрос» или отправьте /ask, затем выберите игру в поиске."
)


async def show_ask_prompt(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(ASK_PROMPT_TEXT, reply_markup=build_inline_search_keyboard())


async def show_cancelled(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Отменено.", reply_markup=build_start_keyboard())


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    name = message.from_user.first_name or "игрок"
    await message.answer(f"Привет, {name}! 👋\n{HELP_TEXT}", reply_markup=build_start_keyboard())


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("cancel"))
@router.message(F.text.casefold() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext) -> None:
    if await state.get_state() is None:
        await message.answer("Нечего отменять.", reply_markup=build_start_keyboard())
        return
    await show_cancelled(message, state)


@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext) -> None:
    await show_ask_prompt(message, state)


@router.callback_query(F.data == CALLBACK_ACTION_ASK)
async def callback_ask(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    if callback.message:
        await show_ask_prompt(callback.message, state)


@router.callback_query(F.data == CALLBACK_ACTION_CANCEL)
async def callback_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer("Отменено.")
    if callback.message:
        await show_cancelled(callback.message, state)


@router.callback_query(F.data == CALLBACK_ACTION_DOWNLOAD_SOURCE)
async def callback_download_source(callback: CallbackQuery, state: FSMContext) -> None:
    await callback.answer()
    data = await state.get_data()
    game_id = data.get("selected_game_id")
    game_title = data.get("selected_game_title") or "Выбранная игра"
    if game_id is None:
        if callback.message:
            await callback.message.answer("Сначала выберите игру через /ask.")
        return

    if callback.message:
        status_message = await callback.message.answer("Готовлю файл правил...")
    else:
        return

    file_bytes, filename, _mime, err = await download_rules_source(int(game_id))
    if err or not file_bytes or not filename:
        await status_message.edit_text(err or "Не удалось скачать файл правил.")
        return

    safe_filename = filename.strip() or f"rules-{game_id}.bin"
    input_file: InputFile = BufferedInputFile(file_bytes, filename=safe_filename)
    await status_message.delete()
    await callback.message.answer_document(
        document=input_file,
        caption=f"Исходник правил для игры: {html.escape(str(game_title))}",
    )


@router.inline_query()
async def process_inline_game_search(inline_query: InlineQuery) -> None:
    search = (inline_query.query or "").strip()
    search_param = search or None
    games = await fetch_games(search=search_param, limit=50)

    results: list[InlineQueryResultArticle] = []
    for g in games:
        gid = g.get("id")
        title = g.get("title", "?")
        if gid is None:
            continue
        safe_title = html.escape(str(title))
        preview = (
            f"Игра: <b>{safe_title}</b>"
            f'<a href="{INLINE_GAME_URL_PREFIX}{gid}">&#8203;</a>'
        )
        results.append(
            InlineQueryResultArticle(
                id=f"g:{gid}",
                title=str(title),
                description="Нажмите, чтобы отправить игру в чат",
                input_message_content=InputTextMessageContent(
                    message_text=preview,
                    parse_mode="HTML",
                ),
            )
        )

    await inline_query.answer(
        results=results,
        is_personal=True,
        cache_time=1,
    )


@router.message(F.via_bot)
async def process_inline_game_message(message: Message, state: FSMContext) -> None:
    text = (message.text or "").strip()
    if not text:
        return

    gid: int | None = None
    for entity in message.entities or []:
        if entity.type != MessageEntityType.TEXT_LINK or not entity.url:
            continue
        if not entity.url.startswith(INLINE_GAME_URL_PREFIX):
            continue
        try:
            gid = int(entity.url.rsplit("/", 1)[1])
        except (ValueError, IndexError):
            return
        break

    if gid is None:
        return

    lines = text.splitlines()
    if not lines:
        return

    first_line = lines[0].strip()
    game_title = first_line.removeprefix("Игра: ").strip() or "Выбранная игра"

    await state.update_data(
        selected_game_id=gid,
        selected_game_title=game_title,
        qa_history=[],
    )
    await state.set_state(AskStates.waiting_question)
    await message.answer(
        f"Игра: <b>{html.escape(game_title)}</b>\n\nЗадайте ваш вопрос:",
        reply_markup=build_waiting_question_keyboard(),
    )


@router.message(AskStates.waiting_question, F.text)
async def process_question_text(message: Message, state: FSMContext) -> None:
    query = (message.text or "").strip()
    if not query:
        await message.answer("Введите вопрос.")
        return

    data = await state.get_data()
    game_id = data.get("selected_game_id")
    qa_history = data.get("qa_history") or []
    if game_id is None:
        await state.clear()
        await message.answer("Ошибка: игра не выбрана. Начните с /ask.")
        return

    history_text = build_history_text(qa_history)
    status_msg = await message.answer("Ищу ответ...")
    answer, err = await ask_question(game_id, query, history=history_text)
    if err:
        await status_msg.edit_text(
            f"{err}\n\nПопробуйте задать вопрос ещё раз или сменить игру.",
            reply_markup=build_waiting_question_keyboard(),
        )
        return

    a = answer or ""
    next_history = [
        *qa_history,
        {
            "q": clip_text(query, settings.max_history_chars_per_item),
            "a": clip_text(a, settings.max_history_chars_per_item),
        },
    ]
    if len(next_history) > settings.max_history_turns:
        next_history = next_history[-settings.max_history_turns :]
    await state.update_data(qa_history=next_history)

    text = a[:4000]
    if len(a) > 4000:
        text += "\n\n(сообщение обрезано)"
    text += "\n\nМожно задать следующий вопрос по этой игре или сменить игру."
    await status_msg.edit_text(text, reply_markup=build_waiting_question_keyboard())


@router.message(AskStates.waiting_question)
async def process_question_non_text(message: Message) -> None:
    """Обработка фото, документов и прочего не-текста в состоянии вопроса."""
    await message.answer("Введите вопрос текстом.")


@router.message(StateFilter(None), F.text)
async def text_without_active_game(message: Message) -> None:
    """Текст вне сессии с выбранной игрой (после отмены, /ask без выбора, сброс FSM и т.п.)."""
    await message.answer(NO_GAME_CONTEXT_TEXT, reply_markup=build_start_keyboard())
