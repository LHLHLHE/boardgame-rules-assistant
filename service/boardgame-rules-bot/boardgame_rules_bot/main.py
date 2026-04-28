import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from boardgame_rules_bot.config import settings
from boardgame_rules_bot.handlers import router
from boardgame_rules_bot.ngrok import resolve_webhook_base_url

logging.basicConfig(level=logging.INFO)

bot = Bot(
    token=settings.bot_token,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher(storage=MemoryStorage())
dp.include_router(router)


async def on_startup(app: web.Application) -> None:
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="открыть главное меню"),
            BotCommand(command="help", description="как пользоваться ботом"),
            BotCommand(command="info", description="список команд"),
            BotCommand(command="ask", description="выбрать игру и задать вопрос"),
            BotCommand(command="cancel", description="отменить текущий сценарий"),
        ]
    )
    base_url = await resolve_webhook_base_url()
    webhook_url = f"{base_url.rstrip('/')}{settings.webhook_path}"
    await bot.set_webhook(
        webhook_url,
        allowed_updates=dp.resolve_used_update_types(),
    )
    logging.info("Webhook URL: %s", webhook_url)


def main() -> None:
    app = web.Application()
    webhook_request_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
    )
    webhook_request_handler.register(app, path=settings.webhook_path)
    setup_application(app, dp, bot=bot)

    app.on_startup.append(on_startup)

    web.run_app(app, host=settings.webapp_host, port=settings.webapp_port)


if __name__ == "__main__":
    main()
