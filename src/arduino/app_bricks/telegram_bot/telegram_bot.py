# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import asyncio
import threading
from arduino.app_utils import brick, Logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

logger = Logger("TelegramBot")


@brick
class TelegramBot:
    """A brick to manage Telegram Bot interactions."""

    def __init__(self, token: str = None):
        """Initialize the bot with a token from arg or environment variable."""
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram TELEGRAM_BOT_TOKEN must be provided or set as environment variable")

        self.application = Application.builder().token(self.token).build()
        self._loop = None
        self._loop_thread = None
        self._running = False

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
        """Send a message to a specific chat ID (synchronous method)."""
        if not self._running or not self._loop:
            logger.error("Bot not started, cannot send message")
            return

        future = asyncio.run_coroutine_threadsafe(self._send_message_async(chat_id, message_text), self._loop)

        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _send_message_async(self, chat_id: int, message_text: str):
        """Internal async method to send a message."""
        logger.info(f"Sending message to chat_id={chat_id}")
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=message_text)
            logger.info("Message sent successfully!")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def start(self):
        """Start the Telegram bot in a background thread."""
        if self._running:
            logger.warning("Bot is already running")
            return

        logger.info("Starting Telegram Bot...")
        self._running = True
        self._loop_thread = threading.Thread(target=self._run_bot, daemon=True)
        self._loop_thread.start()

        # Wait for the bot to be fully initialized
        timeout = 10
        start_time = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        while not self._loop and self._running:
            import time

            time.sleep(0.1)
            if asyncio.get_event_loop().is_running() and asyncio.get_event_loop().time() - start_time > timeout:
                logger.error("Bot initialization timeout")
                break

    def stop(self):
        """Stop the Telegram bot gracefully."""
        if not self._running:
            return

        logger.info("Stopping Telegram Bot...")
        self._running = False

        if self._loop:
            try:
                # Stop the application
                future = asyncio.run_coroutine_threadsafe(self.application.stop(), self._loop)
                future.result(timeout=5)

                # Shutdown the application
                future = asyncio.run_coroutine_threadsafe(self.application.shutdown(), self._loop)
                future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5)
            if self._loop_thread.is_alive():
                logger.warning("Bot thread did not terminate in time")

    def _run_bot(self):
        """Internal method to run the bot's event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self.application.initialize())
            self._loop.run_until_complete(self.application.start())
            self._loop.run_until_complete(self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES))

            # Keep the loop running
            while self._running:
                self._loop.run_until_complete(asyncio.sleep(0.1))

        except Exception as e:
            logger.exception(f"Error in bot event loop: {e}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
            logger.info("Telegram Bot stopped")
