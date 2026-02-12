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
class Sender:
    """Simplified sender and message object with essential attributes.

    This object is passed to user callbacks, containing all the information
    needed to process a message and respond to it. Includes helper methods
    for easy replies without repeating chat_id.

    Attributes:
        chat_id: Telegram chat ID (used to send responses).
        user_id: ID of the user who sent the message.
        user_name: First name of the user.
        username: Username of the user (None if not set).
        text: Text content of the message (None for media).
        image: Photo data as bytearray (None for non-photo messages).
        audio: Audio data as bytearray (None for non-audio messages).
        video: Video data as bytearray (None for non-video messages).
        document: Document data as bytearray (None for non-document messages).
        caption: Media caption text (None if not present).
        message_id: Original message ID (useful for replies).
    """

    # Identification
    chat_id: int
    user_id: int
    user_name: str
    username: Optional[str] = None

    # Message content
    text: Optional[str] = None
    image: Optional[bytearray] = None
    audio: Optional[bytearray] = None
    video: Optional[bytearray] = None
    document: Optional[bytearray] = None
    caption: Optional[str] = None

    # Metadata
    message_id: Optional[int] = None

    # Internal reference for helper methods
    _bot: Optional["TelegramBot"] = None

    def reply(self, text: str) -> bool:
        """Reply to this message with text.

        Args:
            text: Message text to send.

        Returns:
            True if successful, False otherwise.

        Example:
            >>> def handle_text(sender: Sender):
            ...     sender.reply(f"You said: {sender.text}")
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send(self.chat_id, text)

    def reply_photo(self, photo_bytes: bytes, caption: str = "") -> bool:
        """Reply to this message with a photo.

        Args:
            photo_bytes: Photo data as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_photo(self.chat_id, photo_bytes, caption)

    def reply_audio(self, audio_bytes: bytes, caption: str = "") -> bool:
        """Reply to this message with audio.

        Args:
            audio_bytes: Audio data as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_audio(self.chat_id, audio_bytes, caption)

    def reply_video(self, video_bytes: bytes, caption: str = "") -> bool:
        """Reply to this message with video.

        Args:
            video_bytes: Video data as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_video(self.chat_id, video_bytes, caption)

    def reply_document(self, document_bytes: bytes, filename: str = "document", caption: str = "") -> bool:
        """Reply to this message with a document.

        Args:
            document_bytes: Document data as bytes.
            filename: Name for the document file.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_document(self.chat_id, document_bytes, filename, caption)


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
        creates a simplified Sender object, and handles async-to-sync conversion
        automatically. Media files (photo/audio/video/document) are downloaded
        automatically if present.

        User's callback receives only a simplified Sender object, not Update/Context.

        Args:
            callback: User's synchronous callback(sender: Sender) -> None

        Returns:
            Async handler compatible with python-telegram-bot
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            # Extract essential info from Update into simplified Sender object
            sender = Sender(
                chat_id=update.message.chat_id,
                user_id=update.effective_user.id,
                user_name=update.effective_user.first_name,
                username=update.effective_user.username,
                text=update.message.text if update.message.text else None,
                caption=update.message.caption if update.message.caption else None,
                message_id=update.message.message_id,
                _bot=self,
            )

            # Download photo if present
            if update.message.photo:
                try:
                    photo_file = await update.message.photo[-1].get_file()
                    sender.image = await photo_file.download_as_bytearray()
                except Exception as e:
                    logger.error(f"Failed to download photo: {e}")

            # Download audio if present
            if update.message.audio:
                try:
                    audio_file = await update.message.audio.get_file()
                    sender.audio = await audio_file.download_as_bytearray()
                except Exception as e:
                    logger.error(f"Failed to download audio: {e}")

            # Download video if present
            if update.message.video:
                try:
                    video_file = await update.message.video.get_file()
                    sender.video = await video_file.download_as_bytearray()
                except Exception as e:
                    logger.error(f"Failed to download video: {e}")

            # Download document if present
            if update.message.document:
                try:
                    doc_file = await update.message.document.get_file()
                    sender.document = await doc_file.download_as_bytearray()
                except Exception as e:
                    logger.error(f"Failed to download document: {e}")

            # Run user's callback in executor (sync)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, sender)

        return wrapper

    def add_command(self, command: str, callback: Callable[[Sender], None], description: str = "") -> None:
        """Register a command handler (e.g., /start).

        The callback function receives a simplified Sender object containing
        all essential information about the message and user.

        Args:
            command: Command name without '/' (e.g., "start", "hello").
            callback: Function that receives a Sender object.
            description: Optional description shown in Telegram's command menu.

        Example:
            >>> def greet(sender: Sender):
            ...     sender.reply(f"Hello {sender.user_name}!")
            >>> bot.add_command("hello", greet, "Greet the user")
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(CommandHandler(command, handler))

        if description:
            self._commands_registry[command] = description

        logger.info(f"Registered command: /{command}" + (f" - {description}" if description else ""))

    def on_text(self, callback: Callable[[Sender], None]) -> None:
        """Register a handler for text messages.

        The callback function receives a simplified Sender object containing
        the text and user information.

        Args:
            callback: Function that receives a Sender object.

        Example:
            >>> def echo(sender: Sender):
            ...     sender.reply(f"You said: {sender.text}")
            >>> bot.on_text(echo)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handler))
        logger.info("Registered text message handler")

    def on_photo(self, callback: Callable[[Sender], None]) -> None:
        """Register a handler for photo messages.

        The callback function receives a simplified Sender object with the
        photo already downloaded as image (bytearray).

        Args:
            callback: Function that receives a Sender object with image data.

        Example:
            >>> def handle_photo(sender: Sender):
            ...     if sender.image:
            ...         sender.reply("Got your photo!")
            >>> bot.on_photo(handle_photo)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.PHOTO, handler))
        logger.info("Registered photo message handler")

    def on_audio(self, callback: Callable[[Sender], None]) -> None:
        """Register a handler for audio messages.

        The callback function receives a simplified Sender object with the
        audio already downloaded as audio (bytearray).

        Args:
            callback: Function that receives a Sender object with audio data.

        Example:
            >>> def handle_audio(sender: Sender):
            ...     if sender.audio:
            ...         sender.reply("Got your audio!")
            >>> bot.on_audio(handle_audio)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.AUDIO, handler))
        logger.info("Registered audio message handler")

    def on_video(self, callback: Callable[[Sender], None]) -> None:
        """Register a handler for video messages.

        The callback function receives a simplified Sender object with the
        video already downloaded as video (bytearray).

        Args:
            callback: Function that receives a Sender object with video data.

        Example:
            >>> def handle_video(sender: Sender):
            ...     if sender.video:
            ...         sender.reply("Got your video!")
            >>> bot.on_video(handle_video)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.VIDEO, handler))
        logger.info("Registered video message handler")

    def on_document(self, callback: Callable[[Sender], None]) -> None:
        """Register a handler for document messages.

        The callback function receives a simplified Sender object with the
        document already downloaded as document (bytearray).

        Args:
            callback: Function that receives a Sender object with document data.

        Example:
            >>> def handle_document(sender: Sender):
            ...     if sender.document:
            ...         sender.reply("Got your document!")
            >>> bot.on_document(handle_document)
        """
        handler = self._create_message_handler(callback)
        self.application.add_handler(MessageHandler(filters.Document.ALL, handler))
        logger.info("Registered document message handler")

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

    def send_audio(self, chat_id: int, audio_bytes: bytes, caption: str = "") -> bool:
        """Send an audio file to a chat.

        Args:
            chat_id: Telegram chat ID.
            audio_bytes: Audio as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Example:
            >>> with open("audio.mp3", "rb") as f:
            ...     bot.send_audio(123456, f.read(), "Listen to this!")
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send audio")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_audio_async(chat_id, audio_bytes, caption), self._loop)
                future.result(timeout=self.photo_timeout)
                return True
            except TimeoutError:
                logger.warning(f"Audio send timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
            except Exception as e:
                logger.error(f"Failed to send audio: {e}")
                return False

        logger.error(f"Failed to send audio after {self.max_retries} attempts")
        return False

    async def _send_audio_async(self, chat_id: int, audio_bytes: bytes, caption: str) -> None:
        """Internal async method to send audio with network error handling.

        Args:
            chat_id: Telegram chat ID.
            audio_bytes: Audio bytes to send.
            caption: Audio caption.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If audio sending fails for other reasons.
        """
        logger.info(f"Sending audio to chat_id={chat_id}")
        try:
            await self.application.bot.send_audio(
                chat_id=chat_id,
                audio=audio_bytes,
                caption=caption,
                read_timeout=self.photo_timeout,
                write_timeout=self.photo_timeout,
            )
            logger.info("Audio sent successfully!")
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while sending audio: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def send_video(self, chat_id: int, video_bytes: bytes, caption: str = "") -> bool:
        """Send a video to a chat.

        Args:
            chat_id: Telegram chat ID.
            video_bytes: Video as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Example:
            >>> with open("video.mp4", "rb") as f:
            ...     bot.send_video(123456, f.read(), "Check out this video!")
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send video")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_video_async(chat_id, video_bytes, caption), self._loop)
                future.result(timeout=self.photo_timeout)
                return True
            except TimeoutError:
                logger.warning(f"Video send timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
            except Exception as e:
                logger.error(f"Failed to send video: {e}")
                return False

        logger.error(f"Failed to send video after {self.max_retries} attempts")
        return False

    async def _send_video_async(self, chat_id: int, video_bytes: bytes, caption: str) -> None:
        """Internal async method to send video with network error handling.

        Args:
            chat_id: Telegram chat ID.
            video_bytes: Video bytes to send.
            caption: Video caption.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If video sending fails for other reasons.
        """
        logger.info(f"Sending video to chat_id={chat_id}")
        try:
            await self.application.bot.send_video(
                chat_id=chat_id,
                video=video_bytes,
                caption=caption,
                read_timeout=self.photo_timeout,
                write_timeout=self.photo_timeout,
            )
            logger.info("Video sent successfully!")
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while sending video: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise

    def send_document(self, chat_id: int, document_bytes: bytes, filename: str = "document", caption: str = "") -> bool:
        """Send a document to a chat.

        Args:
            chat_id: Telegram chat ID.
            document_bytes: Document as bytes.
            filename: Name for the document file.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Example:
            >>> with open("report.pdf", "rb") as f:
            ...     bot.send_document(123456, f.read(), "report.pdf", "Monthly report")
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send document")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_document_async(chat_id, document_bytes, filename, caption), self._loop)
                future.result(timeout=self.photo_timeout)
                return True
            except TimeoutError:
                logger.warning(f"Document send timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
            except Exception as e:
                logger.error(f"Failed to send document: {e}")
                return False

        logger.error(f"Failed to send document after {self.max_retries} attempts")
        return False

    async def _send_document_async(self, chat_id: int, document_bytes: bytes, filename: str, caption: str) -> None:
        """Internal async method to send document with network error handling.

        Args:
            chat_id: Telegram chat ID.
            document_bytes: Document bytes to send.
            filename: Document filename.
            caption: Document caption.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If document sending fails for other reasons.
        """
        logger.info(f"Sending document to chat_id={chat_id}")
        try:
            await self.application.bot.send_document(
                chat_id=chat_id,
                document=document_bytes,
                filename=filename,
                caption=caption,
                read_timeout=self.photo_timeout,
                write_timeout=self.photo_timeout,
            )
            logger.info("Document sent successfully!")
        except (NetworkError, TimedOut) as e:
            logger.warning(f"Network issue while sending document: {e}")
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
