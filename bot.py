import os
import logging

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Загружаем переменные из .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Список моделей Mistral — бот пробует их по очереди, если одна перегружена
FALLBACK_MODELS = [
    os.getenv("MODEL", "mistral-small-latest"),
    "mistral-medium-latest",
    "open-mistral-7b",
]

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в .env")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY не найден в .env")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Клиент для Mistral AI (использует формат OpenAI SDK)
client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=MISTRAL_API_KEY,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот на базе AI. Просто напиши мне сообщение."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    chat_id = update.effective_chat.id

    reply_text = None
    last_error = None

    # Пробуем модели по очереди. Если одна перегружена (429) — переходим к следующей.
    for model in FALLBACK_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_text},
                ],
            )
            reply_text = response.choices[0].message.content
            logger.info(f"Ответ получен от модели: {model}")
            break
        except Exception as e:
            last_error = e
            logger.warning(f"Модель {model} недоступна ({e}), пробуем следующую")
            continue

    if reply_text is None:
        logger.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
        reply_text = (
            "Сейчас модели Mistral перегружены или закончился баланс. "
            "Попробуй написать ещё раз через минуту."
        )

    await context.bot.send_message(chat_id=chat_id, text=reply_text)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()