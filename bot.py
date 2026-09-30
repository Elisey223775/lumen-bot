import asyncio
import os
import logging
import time

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Загружаем переменные из .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# Универсальный провайдер: по умолчанию Pollinations (без ключа),
# через .env можно переключить на llm7 / Mistral / OpenRouter
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://text.pollinations.ai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("MISTRAL_API_KEY") or "no-key"
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL", "openai-fast")
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "")

# Список моделей — основная + запасная, если одна перегружена
FALLBACK_MODELS = [m for m in [LLM_MODEL, LLM_FALLBACK] if m]
if not FALLBACK_MODELS:
    FALLBACK_MODELS = ["mistral-small-latest"]

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не найден в .env")
# LLM_API_KEY может быть "no-key" для бесплатных endpoint без авторизации

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info(f"LLM: {LLM_BASE_URL} | модель: {LLM_MODEL}")

# Клиент OpenAI-совместимый (Mistral / llm7 / OpenRouter — через .env)
client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)


def load_system_prompt():
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Ты — Lumen, мыслящий собеседник."

SYSTEM_PROMPT = load_system_prompt()
logger.info(f"System prompt загружен: {len(SYSTEM_PROMPT)} символов")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я бот на базе AI. Просто напиши мне сообщение."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text
    chat_id = update.effective_chat.id

    reply_text = None
    last_error = None

    # Пробуем модели по очереди. 429 = лимит провайдера, ждём и пробуем ещё раз.
    for model in FALLBACK_MODELS:
        for attempt in (1, 2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.8,
                    max_tokens=500,
                )
                reply_text = response.choices[0].message.content
                logger.info(f"Ответ получен от модели: {model}")
                break
            except Exception as e:
                last_error = e
                msg = str(e)
                logger.warning(f"Модель {model} попытка {attempt}: {msg[:300]}")
                # 429 — подождать 8 сек и повторить, иначе сразу следующая модель
                if "429" in msg and attempt == 1:
                    await asyncio.sleep(8)
                    continue
                break
        if reply_text:
            break

    if reply_text is None:
        logger.error(f"Все модели недоступны. Последняя ошибка: {last_error}")
        err = str(last_error)[:300] if last_error else "unknown"
        if "429" in err:
            reply_text = (
                "Провайдер упёрся в лимит (429). Подожди минуту и напиши ещё раз."
            )
        elif "401" in err or "403" in err or "Unauthorized" in err:
            reply_text = "Ключ API не подошёл (401/403). Проверь LLM_API_KEY в .env."
        else:
            reply_text = (
                "Сейчас модель недоступна. "
                "Попробуй ещё раз через минуту."
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
