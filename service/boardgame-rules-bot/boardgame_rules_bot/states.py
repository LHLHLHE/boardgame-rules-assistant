from aiogram.fsm.state import State, StatesGroup


class AskStates(StatesGroup):
    waiting_question = State()
