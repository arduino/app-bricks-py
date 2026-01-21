# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import os
import asyncio
import threading
import time
from typing import Callable, Optional
from dataclasses import dataclass
from arduino.app_utils import brick, Logger
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut

logger = Logger("TelegramBot")


@dataclass
class Message:
    """Simplified message object with only essential attributes.

    This object is passed to user callbacks, containing all the information
    needed to process a message and respond to it.

    Attributes:
        chat_id: Telegram chat ID (used to send responses).
        text: Text content of the message (None for photos).
        user_id: ID of the user who sent the message.
        user_name: First name of the user.
        username: Username of the user (None if not set).
        photo_bytes: Photo data as bytes (None for text messages).
    """

    chat_id: int
    text: Optional[str] = None
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    username: Optional[str] = None
    photo_bytes: Optional[bytearray] = None


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
        auto_set_commands: bool = True,
    ) -> None:
        """Initialize the Telegram bot with configurable timeouts and retry settings.

        Args:
            token: Telegram bot token. If not provided, reads from TELEGRAM_BOT_TOKEN
                environment variable.
            message_timeout: Timeout in seconds for sending messages (default: 30).
            photo_timeout: Timeout in seconds for sending/downloading photos (default: 60).
            max_retries: Maximum number of retries for network operations (default: 3).
            auto_set_commands: Automatically sync registered commands with Telegram's
                command menu (default: True). When enabled, commands with descriptions
                will appear when users type '/' in the chat.

        Raises:
            ValueError: If token is not provided and TELEGRAM_BOT_TOKEN env var is not set.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram TELEGRAM_BOT_TOKEN must be provided or set as environment variable")

        self.message_timeout = message_timeout
        self.photo_timeout = photo_timeout
        self.max_retries = max_retries
        self.auto_set_commands = auto_set_commands

        self.application = Application.builder().token(self.token).build()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._initialized: bool = False
        self._scheduled_tasks: dict[str, threading.Timer] = {}
        self._commands_registry: dict[str, str] = {}

    def _create_message_handler(self, callback: Callable) -> Callable:
        """Create a Telegram handler from user's simple callback.

        This method extracts essential information from Telegram's Update object,
        creates a simplified Message object, and handles async-to-sync conversion
        automatically. Photos are downloaded automatically if present.

        User's callback receives only a simplified Message object, not Update/Context.

        Args:
            callback: User's synchronous callback(message: Message) -> None

        Returns:
            Async handler compatible with python-telegram-bot
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            # Extract essential info from Update into simplified Message object
            message = Message(
                chat_id=update.message.chat_id,
                text=update.message.text if update.message.text else None,
                user_id=update.effective_user.id,
                user_name=update.effective_user.first_name,
                username=update.effective_user.username,
            )

            # Download photo if present
            if update.message.photo:
                try:
                    photo_file = await update.message.photo[-1].get_file()
                    message.photo_bytes = await photo_file.download_as_bytearray()
                except Exception as e:
                    logger.error(f"Failed to download photo: {e}")

            # Run user's callback in executor (sync)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, message)

        return wrapper

    def add_command(self, command: str, callback: Callable[[Message], None], description: str = "") -> None:
        """Register a command handler (e.g., /start).

        The callback function receives a simplified Message object containing
        all essential information about the message and user.

        Args:
            command: Command name without '/' (e.g., "start", "hello").
            callback: Function that receives a Message object.
            description: Optional description shown in Telegram's command menu.

        Example:
            >>> def greet(msg: Message):
            ...     bot.send(msg.chat_id, f"Hello {msg.user_name}!")
            >>> bot.add_command("hello", greet, "Greet the user")
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(CommandHandler(command, handler))

        if description:
            self._commands_registry[command] = description

        logger.info(f"Registered command: /{command}" + (f" - {description}" if description else ""))

    def on_text(self, callback: Callable[[Message], None]) -> None:
        """Register a handler for text messages.

        The callback function receives a simplified Message object containing
        the text and user information.

        Args:
            callback: Function that receives a Message object.

        Example:
            >>> def echo(msg: Message):
            ...     bot.send(msg.chat_id, f"You said: {msg.text}")
            >>> bot.on_text(echo)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))
        logger.info("Registered text message handler")

    def on_photo(self, callback: Callable[[Message], None]) -> None:
        """Register a handler for photo messages.

        The callback function receives a simplified Message object with the
        photo already downloaded as photo_bytes.

        Args:
            callback: Function that receives a Message object with photo_bytes.

        Example:
            >>> def handle_photo(msg: Message):
            ...     if msg.photo_bytes:
            ...         bot.send(msg.chat_id, "Got your photo!")
            >>> bot.on_photo(handle_photo)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.PHOTO, handler))
        logger.info("Registered photo message handler")

    def send(self, chat_id: int, text: str) -> bool:
        """Send a text message to a chat (simplified method).

        Args:
            chat_id: Telegram chat ID.
            text: Message text.

        Returns:
            True if message was sent successfully, False otherwise.

        Example:
            >>> bot.send(123456, "Hello from Arduino!")
        """
        return self.send_message(chat_id, text)

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

    def send_photo(self, chat_id: int, photo_bytes: bytes, caption: str = "") -> bool:
        """Send a photo to a chat.

        Args:
            chat_id: Telegram chat ID.
            photo_bytes: Photo as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Example:
            >>> with open("image.jpg", "rb") as f:
            ...     bot.send_photo(123456, f.read(), "Check this out!")
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send photo")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_photo_async(chat_id, photo_bytes, caption), self._loop)
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

    async def _send_photo_async(self, chat_id: int, photo_bytes: bytes, caption: str) -> None:
        """Internal async method to send a photo with network error handling.

        Args:
            chat_id: Telegram chat ID.
            photo_bytes: Photo bytes to send.
            caption: Photo caption.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If photo sending fails for other reasons.
        """
        logger.info(f"Sending photo to chat_id={chat_id}")
        try:
            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=photo_bytes,
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

    def schedule(self, chat_id: int, text: str, interval_seconds: int) -> str:
        """Schedule recurring messages to a chat.

        Args:
            chat_id: Telegram chat ID.
            text: Message text to send.
            interval_seconds: Interval between messages in seconds.

        Returns:
            Task ID that can be used to cancel the scheduled message.

        Example:
            >>> task_id = bot.schedule(123456, "Reminder!", 60)  # Every minute
            >>> # Later: bot.cancel_schedule(task_id)
        """
        return self.schedule_message(chat_id, text, interval_seconds)

    def cancel_schedule(self, task_id: str) -> bool:
        """Cancel a scheduled message.

        Args:
            task_id: Task ID returned by schedule().

        Returns:
            True if task was cancelled, False if task_id not found.
        """
        return self.cancel_scheduled_message(task_id)

    async def _set_bot_commands(self) -> None:
        """Internal method to sync registered commands with Telegram.

        This updates the bot's command menu that appears when users type '/'.
        Only commands with descriptions are registered.

        Raises:
            Exception: If setting commands fails (logged but not raised).
        """
        if not self._commands_registry:
            logger.info("No commands with descriptions to register with Telegram")
            return

        try:
            bot_commands = [BotCommand(command=cmd, description=desc) for cmd, desc in self._commands_registry.items()]
            await self.application.bot.set_my_commands(bot_commands)
            logger.info(f"Successfully registered {len(bot_commands)} command(s) with Telegram's menu")
        except Exception as e:
            logger.error(f"Failed to set bot commands: {e}")

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

            # Auto-register commands with Telegram after polling starts
            if self.auto_set_commands:
                self._loop.run_until_complete(self._set_bot_commands())

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
