from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from boardgame_rules_bot.constants import CALLBACK_ACTION_ASK, CALLBACK_ACTION_CANCEL


def build_start_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Задать вопрос", callback_data=CALLBACK_ACTION_ASK)],
        ]
    )


def build_waiting_question_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔁 Сменить игру",
                    switch_inline_query_current_chat="",
                ),
                InlineKeyboardButton(text="Отмена", callback_data=CALLBACK_ACTION_CANCEL),
            ]
        ]
    )


def build_inline_search_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔍 Найти игру",
                    switch_inline_query_current_chat="",
                )
            ]
        ]
    )
