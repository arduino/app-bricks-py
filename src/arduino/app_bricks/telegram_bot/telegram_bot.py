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
from telegram import Update, BotCommand, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ChatMemberHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut
from .logger_adapter import TelegramLoggerAdapter

logger = Logger("TelegramBot")


@dataclass
class Sender:
    """Represents the sender of a Telegram message.

    Contains user identification and provides convenient methods for replying
    to messages without manually specifying the chat ID.

    Attributes:
        chat_id: Telegram chat ID for sending responses.
        user_id: Unique Telegram user identifier.
        first_name: User's first name.
        last_name: User's last name, None if not set.
        username: User's Telegram username (without @), None if not set.
    """

    chat_id: int
    user_id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None

    # Internal reference for helper methods
    _bot: Optional["TelegramBot"] = None

    def reply(self, text: str) -> bool:
        """Send a text reply to the sender.

        Args:
            text: Text content to send.

        Returns:
            True if message was sent successfully, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_message(self.chat_id, text)

    def reply_photo(self, photo_bytes: bytes, caption: str = "") -> bool:
        """Send a photo reply to the sender.

        Args:
            photo_bytes: Photo data as bytes.
            caption: Optional text caption for the photo.

        Returns:
            True if photo was sent successfully, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_photo(self.chat_id, photo_bytes, caption)

    def reply_audio(self, audio_bytes: bytes, caption: str = "", filename: str = "audio.mp3") -> bool:
        """Send an audio file reply to the sender.

        Args:
            audio_bytes: Audio file data as bytes.
            caption: Optional text caption for the audio.
            filename: Filename with extension, defaults to "audio.mp3".

        Returns:
            True if audio was sent successfully, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_audio(self.chat_id, audio_bytes, caption, filename)

    def reply_video(self, video_bytes: bytes, caption: str = "", filename: str = "video.mp4", supports_streaming: bool = True) -> bool:
        """Send a video file reply to the sender.

        Args:
            video_bytes: Video file data as bytes.
            caption: Optional text caption for the video.
            filename: Filename with extension, defaults to "video.mp4".
            supports_streaming: Enable progressive playback for MP4/H.264 videos.
                Allows playback before full download. Ignored for other formats.

        Returns:
            True if video was sent successfully, False otherwise.

        Note:
            MP4/H.264 videos display inline. Other formats appear as downloadable files.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_video(self.chat_id, video_bytes, caption, filename, supports_streaming)

    def reply_document(self, document_bytes: bytes, filename: str = "document", caption: str = "") -> bool:
        """Send a document file reply to the sender.

        Args:
            document_bytes: Document file data as bytes.
            filename: Filename for the document.
            caption: Optional text caption for the document.

        Returns:
            True if document was sent successfully, False otherwise.
        """
        if not self._bot:
            logger.error("Sender not properly initialized with bot reference")
            return False
        return self._bot.send_document(self.chat_id, document_bytes, filename, caption)


@dataclass
class Message:
    """Represents a Telegram message content and metadata.

    Attributes:
        message_id: Unique message identifier.
        text: Text content, None if message has no text.
        caption: Media caption text, None if no caption present.
    """

    message_id: int
    text: Optional[str] = None
    caption: Optional[str] = None


@brick
class TelegramBot:
    """A brick to manage Telegram Bot interactions with synchronous API.

    Provides a user-friendly interface for creating Telegram bots using synchronous
    methods while handling async operations internally. Includes automatic retries,
    configurable timeouts, and built-in authorization via user ID whitelist.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        message_timeout: int = 30,
        media_timeout: int = 60,
        max_retries: int = 3,
        auto_set_commands: bool = True,
        enable_builtin_welcome: bool = False,
        whitelist_user_ids: Optional[list[int]] = None,
    ) -> None:
        """Initialize the Telegram bot brick.

        Args:
            token: Telegram bot API token. Reads from TELEGRAM_BOT_TOKEN environment
                variable if not provided.
            message_timeout: Timeout in seconds for text messages, defaults to 30.
            media_timeout: Timeout in seconds for media operations, defaults to 60.
            max_retries: Maximum retry attempts for failed operations, defaults to 3.
            auto_set_commands: Automatically sync command menu with Telegram,
                defaults to True.
            enable_builtin_welcome: Automatically register /start command and
                my_chat_member handler to welcome users. Shows user_id, chat_id,
                and first_name. Disabled if user registers custom /start handler.
                Defaults to False.
            whitelist_user_ids: Optional list of authorized Telegram user IDs.
                If provided, only these users can interact with the bot.
                Use @userinfobot on Telegram to get your user ID.

        Raises:
            ValueError: If token not provided and TELEGRAM_BOT_TOKEN not set.

        Note:
            All media files are handled in RAM only. No temporary files written to disk.

            Telegram Bot API limits:
            - Photos: 10 MB max (upload and download)
            - Audio/Video/Documents: 20 MB max (download), 50 MB max (upload)

            Download failures are handled automatically with error messages to users.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("Telegram TELEGRAM_BOT_TOKEN must be provided or set as environment variable")

        self.message_timeout = message_timeout
        self.media_timeout = media_timeout
        self.max_retries = max_retries
        self.auto_set_commands = auto_set_commands
        self.enable_builtin_welcome = enable_builtin_welcome
        self.whitelist_user_ids = whitelist_user_ids

        # Create authorization filter from whitelist if provided
        if self.whitelist_user_ids:
            self._auth_filter = filters.User(user_id=self.whitelist_user_ids)
            logger.info(f"Authorization filter enabled for {len(self.whitelist_user_ids)} user IDs")
        else:
            self._auth_filter = None

        self.application = Application.builder().token(self.token).build()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._initialized: bool = False
        self._scheduled_tasks: dict[str, threading.Timer] = {}
        self._commands_registry: dict[str, str] = {}
        self._welcome_cooldown: dict[int, float] = {}  # Track last welcome message timestamp per user_id

    def _create_text_handler(self, callback: Callable[[Sender, Message], None]) -> Callable:
        """Create a Telegram handler for text messages.

        Args:
            callback: User's callback(sender: Sender, message: Message) -> None

        Returns:
            Async handler compatible with python-telegram-bot
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            sender = Sender(
                chat_id=update.message.chat_id,
                user_id=update.effective_user.id,
                first_name=update.effective_user.first_name,
                last_name=update.effective_user.last_name,
                username=update.effective_user.username,
                _bot=self,
            )

            message = Message(
                message_id=update.message.message_id,
                text=update.message.text,
                caption=None,
            )

            # Run user's callback in executor (sync)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, sender, message)

        return wrapper

    def _create_media_handler(self, callback: Callable[[Sender, Message, bytes, str, int], None], media_type: str) -> Callable:
        """Create a unified Telegram handler for media messages (photo/audio/video/document).

        All media types share the same signature and similar download logic,
        differing only in size checks and Telegram API accessors.

        Args:
            callback: User's callback(sender, message, media_bytes, filename, size) -> None
            media_type: Type of media: "photo", "audio", "video", or "document"

        Returns:
            Async handler compatible with python-telegram-bot
        """

        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            sender = Sender(
                chat_id=update.message.chat_id,
                user_id=update.effective_user.id,
                first_name=update.effective_user.first_name,
                last_name=update.effective_user.last_name,
                username=update.effective_user.username,
                _bot=self,
            )

            message = Message(
                message_id=update.message.message_id,
                text=None,
                caption=update.message.caption,
            )

            # Get media-specific attributes from update
            if media_type == "photo":
                media_obj = update.message.photo[-1]
                filename = "photo.jpg"  # Telegram doesn't provide original photo names
                size = media_obj.file_size
            elif media_type == "audio":
                media_obj = update.message.audio
                filename = media_obj.file_name or "audio.mp3"
                size = media_obj.file_size
            elif media_type == "video":
                media_obj = update.message.video
                filename = media_obj.file_name or "video.mp4"
                size = media_obj.file_size
            elif media_type == "document":
                media_obj = update.message.document
                filename = media_obj.file_name or "document"
                size = media_obj.file_size
            else:
                logger.error(f"Unknown media type: {media_type}")
                return

            # Download media (Telegram enforces size limits automatically)
            log = TelegramLoggerAdapter(logger, user_id=sender.user_id, message_id=message.message_id)
            try:
                media_file = await media_obj.get_file()
                media_bytes = await media_file.download_as_bytearray()
                if size and size > 1024:  # Log only if > 1 KB
                    log.info(f"Downloaded {media_type} '{filename}': {size / 1024:.1f} KB")
            except Exception as e:
                error_msg = f"❌ Errore download '{filename}': {str(e)}"
                await update.message.reply_text(error_msg)
                log.error(f"Failed to download {media_type}: {e}")
                return

            # Success - call user's callback
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, sender, message, media_bytes, filename, size)

        return wrapper

    def add_command(self, command: str, callback: Callable[[Sender, Message], None], description: str = "") -> None:
        """Register a command handler.

        Args:
            command: Command name without leading '/', e.g., 'start'.
            callback: Function to call when command is received. Receives
                Sender and Message objects.
            description: Optional description shown in Telegram's command menu.
                If provided and auto_set_commands is True, appears when user types '/'.
        """
        handler = self._create_text_handler(callback)

        # Apply authorization filter if whitelist is configured
        if self._auth_filter:
            self.application.add_handler(CommandHandler(command, handler, filters=self._auth_filter))
        else:
            self.application.add_handler(CommandHandler(command, handler))

        if description:
            self._commands_registry[command] = description

        logger.info(f"Registered command: /{command}" + (f" - {description}" if description else ""))

    def on_text(self, callback: Callable[[Sender, Message], None]) -> None:
        """Register a handler for all non-command text messages.

        Args:
            callback: Function to call for each text message. Receives
                Sender and Message objects. Does not trigger for commands.
        """
        handler = self._create_text_handler(callback)

        # Build filter with authorization if whitelist is configured
        base_filter = filters.TEXT & ~filters.COMMAND
        final_filter = base_filter & self._auth_filter if self._auth_filter else base_filter

        self.application.add_handler(MessageHandler(final_filter, handler))
        logger.info("Registered text message handler")

    def on_photo(self, callback: Callable[[Sender, Message, bytes, str, int], None]) -> None:
        """Register a handler for photo messages with automatic download.

        Args:
            callback: Function to call when photo is received. Receives:
                - sender: Sender information
                - message: Message metadata
                - photo_bytes: Downloaded photo data
                - filename: Fixed name 'photo.jpg'
                - size: Photo size in bytes

        Note:
            Telegram limit: 10 MB max.
            If download fails, callback is not invoked and error message is sent to user.
        """
        handler = self._create_media_handler(callback, "photo")

        # Build filter with authorization if whitelist is configured
        final_filter = filters.PHOTO & self._auth_filter if self._auth_filter else filters.PHOTO

        self.application.add_handler(MessageHandler(final_filter, handler))
        logger.info("Registered photo message handler")

    def on_audio(self, callback: Callable[[Sender, Message, bytes, str, int], None]) -> None:
        """Register a handler for audio messages with size-checked download.

        Args:
            callback: Function to call when audio is received. Receives:
                - sender: Sender information
                - message: Message metadata
                - audio_bytes: Downloaded audio data
                - filename: Original filename or 'audio.mp3'
                - size: Audio size in bytes

        Note:
            Telegram limit: 20 MB max (download).
            If download fails (size limit or errors), callback is not invoked
            and error message is sent to user.
        """
        handler = self._create_media_handler(callback, "audio")

        # Build filter with authorization if whitelist is configured
        final_filter = filters.AUDIO & self._auth_filter if self._auth_filter else filters.AUDIO

        self.application.add_handler(MessageHandler(final_filter, handler))
        logger.info("Registered audio message handler")

    def on_video(self, callback: Callable[[Sender, Message, bytes, str, int], None]) -> None:
        """Register a handler for video messages with size-checked download.

        Args:
            callback: Function to call when video is received. Receives:
                - sender: Sender information
                - message: Message metadata
                - video_bytes: Downloaded video data
                - filename: Original filename or 'video.mp4'
                - size: Video size in bytes

        Note:
            Telegram limit: 20 MB max (download).
            If download fails (size limit or errors), callback is not invoked
            and error message is sent to user.
        """
        handler = self._create_media_handler(callback, "video")

        # Build filter with authorization if whitelist is configured
        final_filter = filters.VIDEO & self._auth_filter if self._auth_filter else filters.VIDEO

        self.application.add_handler(MessageHandler(final_filter, handler))
        logger.info("Registered video message handler")

    def on_document(self, callback: Callable[[Sender, Message, bytes, str, int], None]) -> None:
        """Register a handler for document messages with size-checked download.

        Args:
            callback: Function to call when document is received. Receives:
                - sender: Sender information
                - message: Message metadata
                - document_bytes: Downloaded document data
                - filename: Original filename or 'document'
                - size: Document size in bytes

        Note:
            Telegram limit: 20 MB max (download).
            If download fails (size limit or errors), callback is not invoked
            and error message is sent to user.
        """
        handler = self._create_media_handler(callback, "document")

        # Build filter with authorization if whitelist is configured
        final_filter = filters.Document.ALL & self._auth_filter if self._auth_filter else filters.Document.ALL

        self.application.add_handler(MessageHandler(final_filter, handler))
        logger.info("Registered document message handler")

    def send_message(self, chat_id: int, message_text: str) -> bool:
        """Send a text message with automatic retry.

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
        log = TelegramLoggerAdapter(logger, chat_id=chat_id)
        log.info("Sending message")
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                read_timeout=self.message_timeout,
                write_timeout=self.message_timeout,
            )
            log.info("Message sent successfully")
        except (NetworkError, TimedOut) as e:
            log.warning(f"Network issue while sending message: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise

    def send_photo(self, chat_id: int, photo_bytes: bytes, caption: str = "") -> bool:
        """Send a photo to a chat.

        Args:
            chat_id: Telegram chat ID.
            photo_bytes: Photo data as bytes.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Note:
            Telegram limit: 10 MB max (upload). Files handled in RAM only, no disk storage.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send photo")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_photo_async(chat_id, photo_bytes, caption), self._loop)
                future.result(timeout=self.media_timeout)
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
        log = TelegramLoggerAdapter(logger, chat_id=chat_id)
        log.info("Sending photo")
        try:
            # Convert bytearray to bytes if needed
            if isinstance(photo_bytes, bytearray):
                photo_bytes = bytes(photo_bytes)

            # Use InputFile to send from memory
            photo = InputFile(photo_bytes)

            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption,
                read_timeout=self.media_timeout,
                write_timeout=self.media_timeout,
            )
            log.info("Photo sent successfully")
        except (NetworkError, TimedOut) as e:
            log.warning(f"Network issue while sending photo: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise

    def send_audio(self, chat_id: int, audio_bytes: bytes, caption: str = "", filename: str = "audio.mp3") -> bool:
        """Send an audio file to a chat.

        Args:
            chat_id: Telegram chat ID.
            audio_bytes: Audio data as bytes.
            caption: Optional caption text.
            filename: Filename with extension, defaults to 'audio.mp3'.
                Extension helps Telegram determine MIME type.
                Supported: .mp3, .m4a, .ogg, etc.

        Returns:
            True if successful, False otherwise.

        Note:
            Telegram limit: 50 MB max (upload). Files handled in RAM only, no disk storage.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send audio")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_audio_async(chat_id, audio_bytes, caption, filename), self._loop)
                future.result(timeout=self.media_timeout)
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

    async def _send_audio_async(self, chat_id: int, audio_bytes: bytes, caption: str, filename: str) -> None:
        """Internal async method to send audio with network error handling.

        Args:
            chat_id: Telegram chat ID.
            audio_bytes: Audio bytes to send.
            caption: Audio caption.
            filename: Filename with extension for MIME type detection.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If audio sending fails for other reasons.
        """
        log = TelegramLoggerAdapter(logger, chat_id=chat_id)
        log.info(f"Sending audio '{filename}'")
        try:
            # Convert bytearray to bytes if needed
            if isinstance(audio_bytes, bytearray):
                audio_bytes = bytes(audio_bytes)

            # Use InputFile to send from memory with filename
            audio = InputFile(audio_bytes, filename=filename)

            await self.application.bot.send_audio(
                chat_id=chat_id,
                audio=audio,
                caption=caption,
                read_timeout=self.media_timeout,
                write_timeout=self.media_timeout,
            )
            log.info("Audio sent successfully")
        except (NetworkError, TimedOut) as e:
            log.warning(f"Network issue while sending audio: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise

    def send_video(self, chat_id: int, video_bytes: bytes, caption: str = "", filename: str = "video.mp4", supports_streaming: bool = True) -> bool:
        """Send a video to a chat.

        Args:
            chat_id: Telegram chat ID.
            video_bytes: Video data as bytes.
            caption: Optional caption text.
            filename: Filename with extension, defaults to 'video.mp4'.
                Extension helps Telegram determine MIME type. Use .mp4 for best compatibility.
            supports_streaming: Enable progressive download for MP4/H.264 videos,
                allowing playback before download completes. Only effective for
                MPEG4 format. Defaults to True.

        Returns:
            True if successful, False otherwise.

        Note:
            Telegram limit: 50 MB max (upload) via multipart/form-data.
            Recommended: MP4 (H.264 video, AAC audio) for inline playback.
            Other formats (AVI, MKV, etc.) sent as downloadable documents.
            Files handled in RAM only, no disk storage.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send video")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_video_async(chat_id, video_bytes, caption, filename, supports_streaming), self._loop
                )
                future.result(timeout=self.media_timeout)
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

    async def _send_video_async(self, chat_id: int, video_bytes: bytes, caption: str, filename: str, supports_streaming: bool) -> None:
        """Internal async method to send video with network error handling.

        Args:
            chat_id: Telegram chat ID.
            video_bytes: Video bytes to send.
            caption: Video caption.
            filename: Filename with extension for MIME type detection.
            supports_streaming: Whether video should support streaming.

        Raises:
            NetworkError: If network issues occur.
            TimedOut: If request times out.
            Exception: If video sending fails for other reasons.
        """
        log = TelegramLoggerAdapter(logger, chat_id=chat_id)
        log.info(f"Sending video '{filename}'")
        try:
            # Convert bytearray to bytes if needed
            if isinstance(video_bytes, bytearray):
                video_bytes = bytes(video_bytes)

            # Use InputFile to send from memory with filename
            video = InputFile(video_bytes, filename=filename)

            await self.application.bot.send_video(
                chat_id=chat_id,
                video=video,
                caption=caption,
                supports_streaming=supports_streaming,
                read_timeout=self.media_timeout,
                write_timeout=self.media_timeout,
            )
            log.info("Video sent successfully")
        except (NetworkError, TimedOut) as e:
            log.warning(f"Network issue while sending video: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise

    def send_document(self, chat_id: int, document_bytes: bytes, filename: str = "document", caption: str = "") -> bool:
        """Send a document to a chat.

        Args:
            chat_id: Telegram chat ID.
            document_bytes: Document data as bytes.
            filename: Document filename. Include extension for proper MIME type detection.
            caption: Optional caption text.

        Returns:
            True if successful, False otherwise.

        Note:
            Telegram limit: 50 MB max (upload). Files handled in RAM only, no disk storage.
        """
        if not self._running or not self._loop or not self._initialized:
            logger.error("Bot not properly initialized, cannot send document")
            return False

        for attempt in range(self.max_retries):
            try:
                future = asyncio.run_coroutine_threadsafe(self._send_document_async(chat_id, document_bytes, filename, caption), self._loop)
                future.result(timeout=self.media_timeout)
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
        log = TelegramLoggerAdapter(logger, chat_id=chat_id)
        log.info(f"Sending document '{filename}'")
        try:
            # Convert bytearray to bytes if needed
            if isinstance(document_bytes, bytearray):
                document_bytes = bytes(document_bytes)

            # Use InputFile to send from memory with filename
            document = InputFile(document_bytes, filename=filename)

            await self.application.bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption,
                read_timeout=self.media_timeout,
                write_timeout=self.media_timeout,
            )
            log.info("Document sent successfully")
        except (NetworkError, TimedOut) as e:
            log.warning(f"Network issue while sending document: {e}")
            raise
        except Exception as e:
            log.error(f"An error occurred: {e}")
            raise

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
        """Schedule a recurring message at regular intervals.

        Args:
            chat_id: Telegram chat ID to send messages to.
            message_text: Text content of the scheduled message.
            interval_seconds: Time interval in seconds between messages.
            task_id: Optional unique identifier for this task.
                If not provided, one is generated automatically.

        Returns:
            Task ID string that can be used to cancel the scheduled message.
            Returns empty string if bot is not initialized.
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
            task_id: ID of the task to cancel, as returned by schedule_message().

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

    def _register_builtin_handlers(self) -> None:
        """Register built-in handlers for /start command and my_chat_member updates.

        Only registers /start if user hasn't already registered a custom handler.
        Always registers my_chat_member to detect when users unblock the bot.
        """
        # Only register /start if user hasn't defined custom handler
        if "start" not in self._commands_registry:

            async def builtin_start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
                """Built-in handler for /start command."""
                user = update.effective_user
                chat_id = update.message.chat_id

                log = TelegramLoggerAdapter(logger, user_id=user.id, chat_id=chat_id)

                # Check cooldown to avoid duplicate messages (e.g., when unblocking triggers both my_chat_member and /start)
                current_time = time.time()
                last_welcome_time = self._welcome_cooldown.get(user.id, 0)
                cooldown_seconds = 3

                if current_time - last_welcome_time < cooldown_seconds:
                    log.info(f"Skipping /start welcome message (cooldown: {current_time - last_welcome_time:.1f}s ago)")
                    return

                welcome_msg = f"👋 Hi {user.first_name}!\n\nThis is your user_id: {user.id}\nThis is your chat_id: {chat_id}"

                log.info("Built-in /start command triggered")
                await update.message.reply_text(welcome_msg)

                # Update cooldown timestamp
                self._welcome_cooldown[user.id] = current_time

            # Apply authorization filter if whitelist configured
            if self._auth_filter:
                self.application.add_handler(CommandHandler("start", builtin_start_handler, filters=self._auth_filter))
            else:
                self.application.add_handler(CommandHandler("start", builtin_start_handler))

            self._commands_registry["start"] = "Get your user ID and chat ID"
            logger.info("Registered built-in /start command handler")

        # Always register my_chat_member handler to detect unblock events
        async def my_chat_member_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Handler for my_chat_member updates (bot blocked/unblocked)."""
            chat_member_update = update.my_chat_member

            # Check if user unblocked the bot (status changed from 'kicked' to 'member')
            old_status = chat_member_update.old_chat_member.status
            new_status = chat_member_update.new_chat_member.status

            if old_status == "kicked" and new_status == "member":
                # User unblocked the bot - send welcome message
                user = chat_member_update.from_user
                chat_id = chat_member_update.chat.id

                log = TelegramLoggerAdapter(logger, user_id=user.id, chat_id=chat_id)

                # Check cooldown to avoid duplicate messages (e.g., when user clicks Start after unblocking)
                current_time = time.time()
                last_welcome_time = self._welcome_cooldown.get(user.id, 0)
                cooldown_seconds = 3

                if current_time - last_welcome_time < cooldown_seconds:
                    log.debug(f"Skipping unblock welcome message (cooldown: {current_time - last_welcome_time:.1f}s ago)")
                    return

                welcome_msg = f"Welcome back {user.first_name}!\nThis is your user_id: {user.id}\nThis is your chat_id: {chat_id}"

                log.info("User unblocked bot - sending welcome message")

                try:
                    await context.bot.send_message(chat_id=chat_id, text=welcome_msg)
                    # Update cooldown timestamp only after successful send
                    self._welcome_cooldown[user.id] = current_time
                except Exception as e:
                    log.error(f"Failed to send welcome message after unblock: {e}")

        self.application.add_handler(ChatMemberHandler(my_chat_member_handler, ChatMemberHandler.MY_CHAT_MEMBER))
        logger.info("Registered built-in my_chat_member handler for unblock detection")

    def start(self) -> None:
        """Start the Telegram bot in a background thread.

        Initializes the bot and starts polling for updates in a separate thread,
        allowing the main application to continue running. Waits for successful
        initialization before returning.

        Raises:
            RuntimeError: If bot fails to initialize within 30 seconds timeout.
        """
        if self._running:
            logger.warning("Bot is already running")
            return

        # Register built-in welcome handlers if enabled
        if self.enable_builtin_welcome:
            self._register_builtin_handlers()

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

        Stops polling, cancels all scheduled messages, shuts down the application,
        and waits for the background thread to terminate.
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
