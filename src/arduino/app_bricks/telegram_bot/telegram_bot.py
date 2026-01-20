# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import asyncio
import threading
import inspect
from typing import Callable, Optional
from arduino.app_utils import brick, Logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logger = Logger("TelegramBot")


@brick
class TelegramBot:
    """A brick to manage Telegram Bot interactions with synchronous API.

    This brick provides a simplified interface to create Telegram bots using
    synchronous methods. It handles the async event loop internally, allowing
    users to write clean, synchronous code while maintaining full bot functionality.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        """Initialize the Telegram bot.

        Args:
            token: Telegram bot token. If not provided, reads from TELEGRAM_BOT_TOKEN
                environment variable.

        Raises:
            ValueError: If token is not provided and TELEGRAM_BOT_TOKEN env var is not set.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram TELEGRAM_BOT_TOKEN must be provided or set as environment variable")

        self.application = Application.builder().token(self.token).build()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False

    def _make_async_handler(self, callback: Callable) -> Callable:
        """Convert a synchronous callback to an async handler if needed.

        Args:
            callback: User-defined callback function (sync or async).

        Returns:
            Async-compatible callback function.
        """
        if inspect.iscoroutinefunction(callback):
            # Already async, use as-is
            return callback

        # Sync callback, wrap it
        async def async_wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, update, context)

        return async_wrapper

    def add_command(self, command: str, callback: Callable) -> None:
        """Register a slash command handler.

        Args:
            command: Command name (without the leading slash, e.g., "start" for /start).
            callback: Handler function (can be sync or async). Receives Update and ContextTypes.
        """
        async_callback = self._make_async_handler(callback)
        self.application.add_handler(CommandHandler(command, async_callback))

    def on_text(self, callback: Callable) -> None:
        """Register a handler for text messages.

        Args:
            callback: Handler function (can be sync or async). Receives Update and ContextTypes.
                Called for all text messages except commands.
        """
        async_callback = self._make_async_handler(callback)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, async_callback))

    def on_photo(self, callback: Callable) -> None:
        """Register a handler for photo messages.

        Args:
            callback: Handler function (can be sync or async). Receives Update and ContextTypes.
                Called when user sends a photo.
        """
        async_callback = self._make_async_handler(callback)
        self.application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, async_callback))

    def send_message(self, chat_id: int, message_text: str) -> None:
        """Send a text message to a specific chat (synchronous).

        Args:
            chat_id: Telegram chat ID to send the message to.
            message_text: Text content of the message.
        """
        if not self._running or not self._loop:
            logger.error("Bot not started, cannot send message")
            return

        future = asyncio.run_coroutine_threadsafe(self._send_message_async(chat_id, message_text), self._loop)

        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _send_message_async(self, chat_id: int, message_text: str) -> None:
        """Internal async method to send a message.

        Args:
            chat_id: Telegram chat ID.
            message_text: Message text.

        Raises:
            Exception: If message sending fails.
        """
        logger.info(f"Sending message to chat_id={chat_id}")
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=message_text)
            logger.info("Message sent successfully!")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def send_photo(self, chat_id: int, photo, caption: Optional[str] = None) -> None:
        """Send a photo to a specific chat (synchronous).

        Args:
            chat_id: Telegram chat ID to send the photo to.
            photo: Photo to send (file path, URL, or file-like object).
            caption: Optional caption for the photo.
        """
        if not self._running or not self._loop:
            logger.error("Bot not started, cannot send photo")
            return

        future = asyncio.run_coroutine_threadsafe(self._send_photo_async(chat_id, photo, caption), self._loop)

        try:
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Failed to send photo: {e}")

    async def _send_photo_async(self, chat_id: int, photo, caption: Optional[str] = None) -> None:
        """Internal async method to send a photo.

        Args:
            chat_id: Telegram chat ID.
            photo: Photo to send.
            caption: Optional caption.

        Raises:
            Exception: If photo sending fails.
        """
        logger.info(f"Sending photo to chat_id={chat_id}")
        try:
            await self.application.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
            logger.info("Photo sent successfully!")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def get_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bytearray]:
        """Download photo from an update (synchronous).

        Args:
            update: Telegram update object containing the photo.
            context: Telegram context object.

        Returns:
            Photo bytes as bytearray, or None if download fails.
        """
        if not self._running or not self._loop:
            logger.error("Bot not started, cannot get photo")
            return None

        future = asyncio.run_coroutine_threadsafe(self._get_photo_async(update, context), self._loop)

        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"Failed to get photo: {e}")
            return None

    async def _get_photo_async(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bytearray:
        """Internal async method to download a photo.

        Args:
            update: Telegram update object.
            context: Telegram context object.

        Returns:
            Photo bytes as bytearray.

        Raises:
            Exception: If photo download fails.
        """
        logger.info("Downloading photo from Telegram...")
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            logger.info("Photo downloaded successfully!")
            return photo_bytes
        except Exception as e:
            logger.error(f"An error occurred while downloading photo: {e}")
            raise

    def start(self) -> None:
        """Start the Telegram bot in a background thread.

        This method initializes the bot and starts polling for updates in a
        separate thread, allowing the main application to continue running.
        """
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

    def stop(self) -> None:
        """Stop the Telegram bot gracefully.

        This method stops the bot polling, shuts down the application, and
        waits for the background thread to terminate.
        """
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

    def _run_bot(self) -> None:
        """Internal method to run the bot's event loop in a background thread."""
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
