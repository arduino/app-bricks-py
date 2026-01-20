# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import asyncio
import threading
import inspect
import time
from typing import Callable, Optional
from arduino.app_utils import brick, Logger
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut

logger = Logger("TelegramBot")


@brick
class TelegramBot:
    """A brick to manage Telegram Bot interactions with synchronous API.

    This brick provides a simplified interface to create Telegram bots using
    synchronous methods. It handles the async event loop internally, allowing
    users to write clean, synchronous code while maintaining full bot functionality.
    Includes automatic retry logic and configurable timeouts for network resilience.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        message_timeout: int = 30,
        photo_timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Telegram bot with configurable timeouts and retry settings.

        Args:
            token: Telegram bot token. If not provided, reads from TELEGRAM_BOT_TOKEN
                environment variable.
            message_timeout: Timeout in seconds for sending messages (default: 30).
            photo_timeout: Timeout in seconds for sending/downloading photos (default: 60).
            max_retries: Maximum number of retries for network operations (default: 3).

        Raises:
            ValueError: If token is not provided and TELEGRAM_BOT_TOKEN env var is not set.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram TELEGRAM_BOT_TOKEN must be provided or set as environment variable")

        self.message_timeout = message_timeout
        self.photo_timeout = photo_timeout
        self.max_retries = max_retries

        self.application = Application.builder().token(self.token).build()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._initialized: bool = False
        self._scheduled_tasks: dict[str, threading.Timer] = {}

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

    def send_message(self, chat_id: int, message_text: str) -> bool:
        """Send a text message to a specific chat (synchronous with automatic retry).

        Args:
            chat_id: Telegram chat ID to send the message to.
            message_text: Text content of the message.

        Returns:
            True if message was sent successfully, False otherwise.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send message")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_message_async(chat_id, message_text), self._loop)
                future.result(timeout=self.message_timeout)
                return True
            except TimeoutError:
                logger.warning(f"Message send timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Simple backoff
                    continue
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                return False

        logger.error(f"Failed to send message after {self.max_retries} attempts")
        return False

    async def _send_message_async(self, chat_id: int, message_text: str) -> None:
        """Internal async method to send a message with network error handling.

        Args:
            chat_id: Telegram chat ID.
            message_text: Message text.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If message sending fails for other reasons.
        """
        logger.info(f"Sending message to chat_id={chat_id}")
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                read_timeout=self.message_timeout,
                write_timeout=self.message_timeout,
            )
            logger.info("Message sent successfully!")
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while sending message: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def send_photo(self, chat_id: int, photo, caption: Optional[str] = None) -> bool:
        """Send a photo to a specific chat (synchronous with automatic retry).

        Args:
            chat_id: Telegram chat ID to send the photo to.
            photo: Photo to send (file path, URL, or file-like object).
            caption: Optional caption for the photo.

        Returns:
            True if photo was sent successfully, False otherwise.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send photo")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_photo_async(chat_id, photo, caption), self._loop)
                future.result(timeout=self.photo_timeout)
                return True
            except TimeoutError:
                logger.warning(f"Photo send timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Longer backoff for photos
                    continue
            except Exception as e:
                logger.error(f"Failed to send photo: {e}")
                return False

        logger.error(f"Failed to send photo after {self.max_retries} attempts")
        return False

    async def _send_photo_async(self, chat_id: int, photo, caption: Optional[str] = None) -> None:
        """Internal async method to send a photo with network error handling.

        Args:
            chat_id: Telegram chat ID.
            photo: Photo to send.
            caption: Optional caption.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If photo sending fails for other reasons.
        """
        logger.info(f"Sending photo to chat_id={chat_id}")
        try:
            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                read_timeout=self.photo_timeout,
                write_timeout=self.photo_timeout,
            )
            logger.info("Photo sent successfully!")
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while sending photo: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def get_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[bytearray]:
        """Download photo from an update (synchronous with automatic retry).

        Args:
            update: Telegram update object containing the photo.
            context: Telegram context object.

        Returns:
            Photo bytes as bytearray, or None if download fails after all retries.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot get photo")
            return None

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._get_photo_async(update, context), self._loop)
                return future.result(timeout=self.photo_timeout)
            except TimeoutError:
                logger.warning(f"Photo download timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Backoff for downloads
                    continue
            except Exception as e:
                logger.error(f"Failed to get photo: {e}")
                return None

        logger.error(f"Failed to download photo after {self.max_retries} attempts")
        return None

    async def _get_photo_async(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bytearray:
        """Internal async method to download a photo with network error handling.

        Args:
            update: Telegram update object.
            context: Telegram context object.

        Returns:
            Photo bytes as bytearray.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If photo download fails for other reasons.
        """
        logger.info("Downloading photo from Telegram...")
        try:
            photo_file = await update.message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            logger.info("Photo downloaded successfully!")
            return photo_bytes
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while downloading photo: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while downloading photo: {e}")
            raise

    def schedule_message(
        self,
        chat_id: int,
        message_text: str,
        interval_seconds: int,
        task_id: Optional[str] = None,
    ) -> str:
        """Schedule a recurring message to be sent at regular intervals.

        Args:
            chat_id: Telegram chat ID to send messages to.
            message_text: Text content of the scheduled message.
            interval_seconds: Time interval in seconds between messages.
            task_id: Optional unique identifier for this task. If not provided,
                one will be generated automatically.

        Returns:
            Task ID that can be used to cancel the scheduled message.

        Example:
            >>> task_id = bot.schedule_message(123456, "Hello!", 60)
            >>> # Cancel later:
            >>> bot.cancel_scheduled_message(task_id)
        """
        if not self._running or not self._initialized:
            logger.error("Bot not properly initialized, cannot schedule message")
            return ""

        # Generate task_id if not provided
        if task_id is None:
            task_id = f"schedule_{chat_id}_{int(time.time())}"

        def send_and_reschedule():
            """Send message and schedule next occurrence."""
            if not self._running:
                return

            # Send the message
            success = self.send_message(chat_id, message_text)
            if success:
                logger.info(f"Scheduled message sent to chat_id={chat_id}")
            else:
                logger.warning(f"Failed to send scheduled message to chat_id={chat_id}")

            # Reschedule if still running
            if self._running and task_id in self._scheduled_tasks:
                timer = threading.Timer(interval_seconds, send_and_reschedule)
                timer.daemon = True
                self._scheduled_tasks[task_id] = timer
                timer.start()

        # Start the first timer
        timer = threading.Timer(interval_seconds, send_and_reschedule)
        timer.daemon = True
        self._scheduled_tasks[task_id] = timer
        timer.start()

        logger.info(f"Scheduled message task '{task_id}' created (interval: {interval_seconds}s)")
        return task_id

    def cancel_scheduled_message(self, task_id: str) -> bool:
        """Cancel a scheduled message task.

        Args:
            task_id: ID of the task to cancel (returned by schedule_message).

        Returns:
            True if task was found and cancelled, False otherwise.
        """
        if task_id in self._scheduled_tasks:
            timer = self._scheduled_tasks.pop(task_id)
            timer.cancel()
            logger.info(f"Cancelled scheduled message task '{task_id}'")
            return True

        logger.warning(f"Scheduled message task '{task_id}' not found")
        return False

    def start(self) -> None:
        """Start the Telegram bot in a background thread with initialization check.

        This method initializes the bot and starts polling for updates in a
        separate thread, allowing the main application to continue running.
        Waits for successful initialization before returning.

        Raises:
            RuntimeError: If bot fails to initialize within timeout (30 seconds).
        """
        if self._running:
            logger.warning("Bot is already running")
            return

        logger.info("Starting Telegram Bot...")
        self._running = True
        self._initialized = False
        self._loop_thread = threading.Thread(target=self._run_bot, daemon=True)
        self._loop_thread.start()

        # Wait for the bot to be fully initialized
        timeout = 30
        start = time.time()
        while not self._initialized and self._running:
            time.sleep(0.2)
            if time.time() - start > timeout:
                self._running = False
                logger.error("Bot initialization timeout")
                raise RuntimeError("Telegram bot failed to initialize within timeout")

        if not self._initialized:
            raise RuntimeError("Telegram bot initialization failed")

        logger.info("Telegram bot initialized successfully")

    def stop(self) -> None:
        """Stop the Telegram bot gracefully.

        This method stops the bot polling, shuts down the application, and
        waits for the background thread to terminate.
        """
        if not self._running:
            return

        logger.info("Stopping Telegram Bot...")
        self._running = False

        # Cancel all scheduled messages
        for task_id in list(self._scheduled_tasks.keys()):
            self.cancel_scheduled_message(task_id)

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

            self._initialized = True  # Signal successful initialization
            logger.info("Bot polling started successfully")

            # Keep the loop running
            while self._running:
                self._loop.run_until_complete(asyncio.sleep(0.1))

        except Exception as e:
            logger.exception(f"Error in bot event loop: {e}")
            self._running = False
            self._initialized = False
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None
            logger.info("Telegram Bot stopped")
