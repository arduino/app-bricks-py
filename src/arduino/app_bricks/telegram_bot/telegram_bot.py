# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
from arduino.app_utils import brick, Logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import requests

logger = Logger("TelegramBot")

@brick
class TelegramBot:
    """A brick to manage Telegram Bot interactions."""

    def __init__(self, token: str = None):
        """Initialize the bot with a token from arg or environment variable."""
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram BOT_TOKEN must be provided or set as environment variable")

        self.application = Application.builder().token(self.token).build()

    def add_command(self, command: str, callback):
        """Register a slash command (e.g., /start)."""
        self.application.add_handler(CommandHandler(command, callback))

    def on_text(self, callback):
        """Register a handler for text messages."""
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, callback))

    def on_photo(self, callback):
        """Register a handler for photo messages."""
        self.application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, callback))

    def send_message(self, chat_id: int, message_text: str):
        """Send a message to a specific chat ID."""
        logger.info(f"Sending message to chat_id={chat_id if chat_id != 0 else self.chat_id}")
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message_text
        }
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()   
            logger.info("Message sent successfully!")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred: {e}")
            return None

    def run(self):
        """Start the Telegram polling loop."""
        logger.info("Telegram Bot starting...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
