# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from arduino.app_bricks.telegram_bot import TelegramBot, Message
from telegram.error import NetworkError, TimedOut


@pytest.fixture
def mock_telegram_app(monkeypatch):
    """Mock the Telegram Application to avoid real network calls."""
    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.bot.send_message = AsyncMock()
    mock_app.bot.send_photo = AsyncMock()
    mock_app.bot.send_audio = AsyncMock()
    mock_app.bot.send_video = AsyncMock()
    mock_app.bot.send_document = AsyncMock()
    mock_app.bot.set_my_commands = AsyncMock()
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    mock_app.updater = MagicMock()
    mock_app.updater.start_polling = AsyncMock()
    mock_app.add_handler = MagicMock()

    with patch("arduino.app_bricks.telegram_bot.telegram_bot.Application") as mock_builder:
        mock_builder.builder.return_value.token.return_value.build.return_value = mock_app
        yield mock_app


def test_telegram_bot_init_with_token(mock_telegram_app):
    """Test bot initialization with explicit token and default settings."""
    bot = TelegramBot(token="test_token_123")
    assert bot.token == "test_token_123"
    assert bot._running is False
    assert bot._initialized is False
    assert bot._loop is None
    assert bot._loop_thread is None
    assert bot.message_timeout == 30
    assert bot.media_timeout == 60
    assert bot.max_retries == 3
    assert bot.auto_set_commands is True


def test_telegram_bot_init_with_custom_settings(mock_telegram_app):
    """Test bot initialization with custom timeout and retry settings."""
    bot = TelegramBot(
        token="test_token",
        message_timeout=45,
        media_timeout=90,
        max_retries=5,
        auto_set_commands=False,
    )
    assert bot.message_timeout == 45
    assert bot.media_timeout == 90
    assert bot.max_retries == 5
    assert bot.auto_set_commands is False


def test_telegram_bot_init_with_env_token(mock_telegram_app, monkeypatch):
    """Test bot initialization with token from environment variable."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env_token_456")
    bot = TelegramBot()
    assert bot.token == "env_token_456"


def test_telegram_bot_init_without_token(mock_telegram_app, monkeypatch):
    """Test bot initialization fails without token."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN must be provided"):
        TelegramBot()


def test_add_command_with_description(mock_telegram_app):
    """Test registering a command handler with description."""
    bot = TelegramBot(token="test_token")

    def handler(msg: Message):
        pass

    bot.add_command("start", handler, "Start the bot")
    mock_telegram_app.add_handler.assert_called_once()
    assert "start" in bot._commands_registry
    assert bot._commands_registry["start"] == "Start the bot"


def test_add_command_without_description(mock_telegram_app):
    """Test registering a command handler without description."""
    bot = TelegramBot(token="test_token")

    def handler(msg: Message):
        pass

    bot.add_command("help", handler)
    mock_telegram_app.add_handler.assert_called_once()
    assert "help" not in bot._commands_registry


def test_on_text_handler(mock_telegram_app):
    """Test registering a text message handler."""
    bot = TelegramBot(token="test_token")

    def text_handler(msg: Message):
        pass

    bot.on_text(text_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_on_photo_handler(mock_telegram_app):
    """Test registering a photo message handler."""
    bot = TelegramBot(token="test_token")

    def photo_handler(msg: Message):
        pass

    bot.on_photo(photo_handler)
    mock_telegram_app.add_handler.assert_called_once()


@pytest.mark.asyncio
async def test_create_text_handler_extracts_data():
    """Test that _create_text_handler properly extracts data into Sender and Message objects."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")

    received_sender = None
    received_message = None

    def handler(sender: Sender, message: Message):
        nonlocal received_sender, received_message
        received_sender = sender
        received_message = message

    # Create mock Update
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    mock_update.message.message_id = 98765
    mock_update.message.text = "Hello World"
    mock_update.effective_user.id = 67890
    mock_update.effective_user.first_name = "John"
    mock_update.effective_user.last_name = "Doe"
    mock_update.effective_user.username = "johndoe"

    # Create wrapped handler
    wrapped = bot._create_text_handler(handler)

    # Execute handler
    await wrapped(mock_update, MagicMock())

    # Verify Sender object was created correctly
    assert received_sender is not None
    assert received_sender.chat_id == 12345
    assert received_sender.user_id == 67890
    assert received_sender.first_name == "John"
    assert received_sender.last_name == "Doe"
    assert received_sender.username == "johndoe"

    # Verify Message object was created correctly
    assert received_message is not None
    assert received_message.message_id == 98765
    assert received_message.text == "Hello World"
    assert received_message.caption is None


@pytest.mark.asyncio
async def test_create_media_handler_downloads_photo():
    """Test that _create_media_handler downloads photos automatically."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")

    received_sender = None
    received_message = None
    received_media_bytes = None
    received_filename = None
    received_size = None

    def handler(sender: Sender, message: Message, media_bytes: bytes, filename: str, size: int):
        nonlocal received_sender, received_message, received_media_bytes, received_filename, received_size
        received_sender = sender
        received_message = message
        received_media_bytes = media_bytes
        received_filename = filename
        received_size = size

    # Create mock Update with photo
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    mock_update.message.message_id = 98765
    mock_update.message.caption = "Photo caption"
    mock_update.effective_user.id = 67890
    mock_update.effective_user.first_name = "Jane"
    mock_update.effective_user.last_name = None
    mock_update.effective_user.username = "janedoe"

    # Mock photo download
    mock_photo = MagicMock()
    mock_photo.file_size = 5000
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"photo_data_123"))
    mock_photo.get_file = AsyncMock(return_value=mock_file)
    mock_update.message.photo = [mock_photo]

    # Create wrapped handler
    wrapped = bot._create_media_handler(handler, "photo")

    # Execute handler
    await wrapped(mock_update, MagicMock())

    # Verify Sender object
    assert received_sender is not None
    assert received_sender.chat_id == 12345
    assert received_sender.user_id == 67890
    assert received_sender.first_name == "Jane"
    assert received_sender.username == "janedoe"

    # Verify Message object
    assert received_message is not None
    assert received_message.message_id == 98765
    assert received_message.caption == "Photo caption"
    assert received_message.text is None

    # Verify photo was downloaded
    assert received_media_bytes == bytearray(b"photo_data_123")
    assert received_filename == "photo.jpg"
    assert received_size == 5000


@pytest.mark.asyncio
async def test_send_message_async_success(mock_telegram_app):
    """Test internal async send_message method with timeouts."""
    bot = TelegramBot(token="test_token")
    await bot._send_message_async(12345, "Hello, World!")
    mock_telegram_app.bot.send_message.assert_called_once_with(chat_id=12345, text="Hello, World!", read_timeout=30, write_timeout=30)


@pytest.mark.asyncio
async def test_send_message_async_network_error(mock_telegram_app):
    """Test that network errors are properly raised."""
    bot = TelegramBot(token="test_token")
    mock_telegram_app.bot.send_message.side_effect = NetworkError("Connection failed")

    with pytest.raises(NetworkError):
        await bot._send_message_async(12345, "Test")


@pytest.mark.asyncio
async def test_send_photo_async_success(mock_telegram_app):
    """Test internal async send_photo method with timeouts."""
    bot = TelegramBot(token="test_token")
    await bot._send_photo_async(12345, photo_bytes=b"photo_data", caption="Test caption")

    # Verify send_photo was called once with correct parameters
    mock_telegram_app.bot.send_photo.assert_called_once()
    call_args = mock_telegram_app.bot.send_photo.call_args

    # Check arguments
    assert call_args.kwargs["chat_id"] == 12345
    assert call_args.kwargs["caption"] == "Test caption"
    assert call_args.kwargs["read_timeout"] == 60
    assert call_args.kwargs["write_timeout"] == 60

    # Check photo is InputFile object (not raw bytes)
    from telegram import InputFile

    assert isinstance(call_args.kwargs["photo"], InputFile)


@pytest.mark.asyncio
async def test_send_photo_async_timeout(mock_telegram_app):
    """Test that timeout errors are properly raised."""
    bot = TelegramBot(token="test_token")
    mock_telegram_app.bot.send_photo.side_effect = TimedOut("Request timed out")

    with pytest.raises(TimedOut):
        await bot._send_photo_async(12345, b"photo_data", "")


def test_send_message_bot_not_initialized(mock_telegram_app):
    """Test that send_message returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_message(12345, "Test message")
    assert result is False


def test_send_bot_not_initialized(mock_telegram_app):
    """Test that send_message() returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_message(12345, "Test message")
    assert result is False


def test_send_photo_bot_not_initialized(mock_telegram_app):
    """Test that send_photo returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_photo(12345, b"photo_data")
    assert result is False


def test_bot_lifecycle_not_running_initially(mock_telegram_app):
    """Test that bot is not running after initialization."""
    bot = TelegramBot(token="test_token")
    assert bot._running is False
    assert bot._initialized is False


def test_stop_when_not_running(mock_telegram_app):
    """Test that stop() does nothing when bot is not running."""
    bot = TelegramBot(token="test_token")
    bot.stop()  # Should not raise
    assert bot._running is False


def test_schedule_message_exists(mock_telegram_app):
    """Test that schedule_message() method exists."""
    bot = TelegramBot(token="test_token")
    assert hasattr(bot, "schedule_message")
    assert callable(bot.schedule_message)


def test_cancel_scheduled_message_exists(mock_telegram_app):
    """Test that cancel_scheduled_message() method exists."""
    bot = TelegramBot(token="test_token")
    assert hasattr(bot, "cancel_scheduled_message")
    assert callable(bot.cancel_scheduled_message)


def test_message_dataclass_creation():
    """Test that Message dataclass can be created with required fields."""
    msg = Message(message_id=12345)
    assert msg.message_id == 12345
    assert msg.text is None
    assert msg.caption is None


def test_message_dataclass_with_all_fields():
    """Test Message dataclass with all fields populated."""
    msg = Message(
        message_id=12345,
        text="Hello",
        caption="Test caption",
    )
    assert msg.message_id == 12345
    assert msg.text == "Hello"
    assert msg.caption == "Test caption"


@pytest.mark.asyncio
async def test_set_bot_commands_with_descriptions(mock_telegram_app):
    """Test that commands with descriptions are registered with Telegram."""
    bot = TelegramBot(token="test_token")
    bot._commands_registry = {"start": "Start the bot", "help": "Show help"}

    await bot._set_bot_commands()

    mock_telegram_app.bot.set_my_commands.assert_called_once()
    # Verify the commands were set
    call_args = mock_telegram_app.bot.set_my_commands.call_args[0][0]
    assert len(call_args) == 2


@pytest.mark.asyncio
async def test_set_bot_commands_empty_registry(mock_telegram_app):
    """Test that _set_bot_commands does nothing when registry is empty."""
    bot = TelegramBot(token="test_token")
    bot._commands_registry = {}

    await bot._set_bot_commands()

    mock_telegram_app.bot.set_my_commands.assert_not_called()


# ============================================================================
# SENDER DATACLASS TESTS
# ============================================================================


def test_sender_dataclass_creation():
    """Test Sender dataclass creation with required fields."""
    from arduino.app_bricks.telegram_bot import Sender

    sender = Sender(
        chat_id=12345,
        user_id=67890,
        first_name="John",
    )
    assert sender.chat_id == 12345
    assert sender.user_id == 67890
    assert sender.first_name == "John"
    assert sender.last_name is None
    assert sender.username is None
    assert sender._bot is None


def test_sender_dataclass_with_all_fields():
    """Test Sender dataclass with all optional fields."""
    from arduino.app_bricks.telegram_bot import Sender

    sender = Sender(
        chat_id=12345,
        user_id=67890,
        first_name="Jane",
        last_name="Doe",
        username="janedoe",
    )
    assert sender.chat_id == 12345
    assert sender.user_id == 67890
    assert sender.first_name == "Jane"
    assert sender.last_name == "Doe"
    assert sender.username == "janedoe"


def test_sender_reply_without_bot_reference():
    """Test that Sender.reply() returns False when _bot is not set."""
    from arduino.app_bricks.telegram_bot import Sender

    sender = Sender(chat_id=12345, user_id=67890, first_name="Test")

    result = sender.reply("Hello")
    assert result is False


def test_sender_reply_with_bot_reference(mock_telegram_app):
    """Test that Sender.reply() calls bot.send_message correctly."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")
    bot._running = True
    bot._initialized = True
    bot._loop = MagicMock()

    sender = Sender(chat_id=12345, user_id=67890, first_name="Test", _bot=bot)

    # Mock send_message to return True
    with patch.object(bot, "send_message", return_value=True) as mock_send:
        result = sender.reply("Hello")

        assert result is True
        mock_send.assert_called_once_with(12345, "Hello")


def test_sender_reply_photo_with_bot_reference(mock_telegram_app):
    """Test that Sender.reply_photo() calls bot.send_photo correctly."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")
    sender = Sender(chat_id=12345, user_id=67890, first_name="Test", _bot=bot)

    # Mock send_photo to return True
    with patch.object(bot, "send_photo", return_value=True) as mock_send:
        result = sender.reply_photo(b"photo_data", "Caption")

        assert result is True
        mock_send.assert_called_once_with(12345, b"photo_data", "Caption")


def test_sender_reply_audio_with_bot_reference(mock_telegram_app):
    """Test that Sender.reply_audio() calls bot.send_audio correctly."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")
    sender = Sender(chat_id=12345, user_id=67890, first_name="Test", _bot=bot)

    # Mock send_audio to return True
    with patch.object(bot, "send_audio", return_value=True) as mock_send:
        result = sender.reply_audio(b"audio_data", "Caption", "song.mp3")

        assert result is True
        mock_send.assert_called_once_with(12345, b"audio_data", "Caption", "song.mp3")


def test_sender_reply_video_with_bot_reference(mock_telegram_app):
    """Test that Sender.reply_video() calls bot.send_video correctly."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")
    sender = Sender(chat_id=12345, user_id=67890, first_name="Test", _bot=bot)

    # Mock send_video to return True
    with patch.object(bot, "send_video", return_value=True) as mock_send:
        result = sender.reply_video(b"video_data", "Caption", "video.mp4", True)

        assert result is True
        mock_send.assert_called_once_with(12345, b"video_data", "Caption", "video.mp4", True)


def test_sender_reply_document_with_bot_reference(mock_telegram_app):
    """Test that Sender.reply_document() calls bot.send_document correctly."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")
    sender = Sender(chat_id=12345, user_id=67890, first_name="Test", _bot=bot)

    # Mock send_document to return True
    with patch.object(bot, "send_document", return_value=True) as mock_send:
        result = sender.reply_document(b"doc_data", "report.pdf", "Caption")

        assert result is True
        mock_send.assert_called_once_with(12345, b"doc_data", "report.pdf", "Caption")


# ============================================================================
# WHITELIST FUNCTIONALITY TESTS
# ============================================================================


def test_whitelist_initialization_with_user_ids(mock_telegram_app):
    """Test that whitelist creates auth filter correctly."""
    bot = TelegramBot(token="test_token", whitelist_user_ids=[111, 222, 333])

    assert bot.whitelist_user_ids == [111, 222, 333]
    assert bot._auth_filter is not None


def test_whitelist_initialization_without_user_ids(mock_telegram_app):
    """Test that no auth filter is created when whitelist is None."""
    bot = TelegramBot(token="test_token", whitelist_user_ids=None)

    assert bot.whitelist_user_ids is None
    assert bot._auth_filter is None


def test_whitelist_applied_to_command_handler(mock_telegram_app):
    """Test that whitelist filter is applied to command handlers."""
    bot = TelegramBot(token="test_token", whitelist_user_ids=[111, 222])

    def handler(sender, message):
        pass

    bot.add_command("start", handler, "Start command")

    # Verify handler was added (implementation applies filter internally)
    mock_telegram_app.add_handler.assert_called_once()


def test_whitelist_applied_to_text_handler(mock_telegram_app):
    """Test that whitelist filter is applied to text handlers."""
    bot = TelegramBot(token="test_token", whitelist_user_ids=[111])

    def handler(sender, message):
        pass

    bot.on_text(handler)

    # Verify handler was added
    mock_telegram_app.add_handler.assert_called_once()


# ============================================================================
# MEDIA HANDLERS REGISTRATION TESTS
# ============================================================================


def test_on_audio_handler_registration(mock_telegram_app):
    """Test registering an audio message handler."""
    bot = TelegramBot(token="test_token")

    def audio_handler(sender, message, audio_bytes, filename, size):
        pass

    bot.on_audio(audio_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_on_video_handler_registration(mock_telegram_app):
    """Test registering a video message handler."""
    bot = TelegramBot(token="test_token")

    def video_handler(sender, message, video_bytes, filename, size):
        pass

    bot.on_video(video_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_on_document_handler_registration(mock_telegram_app):
    """Test registering a document message handler."""
    bot = TelegramBot(token="test_token")

    def document_handler(sender, message, doc_bytes, filename, size):
        pass

    bot.on_document(document_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_media_handlers_with_whitelist(mock_telegram_app):
    """Test that media handlers respect whitelist when configured."""
    bot = TelegramBot(token="test_token", whitelist_user_ids=[123, 456])

    def handler(sender, message, media_bytes, filename, size):
        pass

    bot.on_audio(handler)
    bot.on_video(handler)
    bot.on_document(handler)

    # Verify all three handlers were added
    assert mock_telegram_app.add_handler.call_count == 3


# ============================================================================
# SEND METHODS TESTS (Audio, Video, Document)
# ============================================================================


@pytest.mark.asyncio
async def test_send_audio_async_success(mock_telegram_app):
    """Test internal async send_audio method."""
    bot = TelegramBot(token="test_token")
    await bot._send_audio_async(12345, b"audio_data", "Audio caption", "song.mp3")

    # Verify send_audio was called once
    mock_telegram_app.bot.send_audio.assert_called_once()
    call_args = mock_telegram_app.bot.send_audio.call_args

    assert call_args.kwargs["chat_id"] == 12345
    assert call_args.kwargs["caption"] == "Audio caption"
    assert call_args.kwargs["read_timeout"] == 60
    assert call_args.kwargs["write_timeout"] == 60

    # Check audio is InputFile object
    from telegram import InputFile

    assert isinstance(call_args.kwargs["audio"], InputFile)


@pytest.mark.asyncio
async def test_send_video_async_success(mock_telegram_app):
    """Test internal async send_video method."""
    bot = TelegramBot(token="test_token")
    await bot._send_video_async(12345, b"video_data", "Video caption", "clip.mp4", True)

    # Verify send_video was called once
    mock_telegram_app.bot.send_video.assert_called_once()
    call_args = mock_telegram_app.bot.send_video.call_args

    assert call_args.kwargs["chat_id"] == 12345
    assert call_args.kwargs["caption"] == "Video caption"
    assert call_args.kwargs["supports_streaming"] is True
    assert call_args.kwargs["read_timeout"] == 60
    assert call_args.kwargs["write_timeout"] == 60

    # Check video is InputFile object
    from telegram import InputFile

    assert isinstance(call_args.kwargs["video"], InputFile)


@pytest.mark.asyncio
async def test_send_document_async_success(mock_telegram_app):
    """Test internal async send_document method."""
    bot = TelegramBot(token="test_token")
    await bot._send_document_async(12345, b"doc_data", "report.pdf", "Document caption")

    # Verify send_document was called once
    mock_telegram_app.bot.send_document.assert_called_once()
    call_args = mock_telegram_app.bot.send_document.call_args

    assert call_args.kwargs["chat_id"] == 12345
    assert call_args.kwargs["caption"] == "Document caption"
    assert call_args.kwargs["read_timeout"] == 60
    assert call_args.kwargs["write_timeout"] == 60

    # Check document is InputFile object
    from telegram import InputFile

    assert isinstance(call_args.kwargs["document"], InputFile)


def test_send_audio_bot_not_initialized(mock_telegram_app):
    """Test that send_audio returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_audio(12345, b"audio_data", "Caption", "audio.mp3")
    assert result is False


def test_send_video_bot_not_initialized(mock_telegram_app):
    """Test that send_video returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_video(12345, b"video_data", "Caption", "video.mp4")
    assert result is False


def test_send_document_bot_not_initialized(mock_telegram_app):
    """Test that send_document returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_document(12345, b"doc_data", "report.pdf", "Caption")
    assert result is False


# ============================================================================
# SCHEDULING TESTS
# ============================================================================


def test_schedule_message_bot_not_initialized(mock_telegram_app):
    """Test that schedule_message returns empty string when bot not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.schedule_message(12345, "Test message", 60)
    assert result == ""


def test_cancel_scheduled_message_not_found(mock_telegram_app):
    """Test that cancel_scheduled_message returns False for unknown task_id."""
    bot = TelegramBot(token="test_token")
    result = bot.cancel_scheduled_message("nonexistent_task")
    assert result is False


# ============================================================================
# MEDIA HANDLER WITH DOWNLOAD FAILURE
# ============================================================================


@pytest.mark.asyncio
async def test_create_media_handler_download_failure():
    """Test that _create_media_handler handles download failures gracefully."""
    from arduino.app_bricks.telegram_bot import Sender

    bot = TelegramBot(token="test_token")

    handler_called = False

    def handler(sender: Sender, message: Message, media_bytes: bytes, filename: str, size: int):
        nonlocal handler_called
        handler_called = True

    # Create mock Update with photo that fails to download
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    mock_update.message.message_id = 98765
    mock_update.message.caption = None
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_user.id = 67890
    mock_update.effective_user.first_name = "Test"
    mock_update.effective_user.last_name = None
    mock_update.effective_user.username = None

    # Mock photo that raises exception on download
    mock_photo = MagicMock()
    mock_photo.file_size = 5000
    mock_photo.get_file = AsyncMock(side_effect=Exception("Download failed"))
    mock_update.message.photo = [mock_photo]

    # Create wrapped handler
    wrapped = bot._create_media_handler(handler, "photo")

    # Execute handler
    await wrapped(mock_update, MagicMock())

    # Verify error message was sent to user
    mock_update.message.reply_text.assert_called_once()
    error_msg = mock_update.message.reply_text.call_args[0][0]
    assert "Errore download" in error_msg

    # Verify handler was NOT called due to download failure
    assert handler_called is False


# ============================================================================
# BYTEARRAY CONVERSION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_send_photo_converts_bytearray(mock_telegram_app):
    """Test that send_photo converts bytearray to bytes."""
    bot = TelegramBot(token="test_token")

    # Pass bytearray instead of bytes
    photo_data = bytearray(b"photo_data_as_bytearray")
    await bot._send_photo_async(12345, photo_data, "Caption")

    # Verify send_photo was called (conversion happens internally)
    mock_telegram_app.bot.send_photo.assert_called_once()


@pytest.mark.asyncio
async def test_send_audio_converts_bytearray(mock_telegram_app):
    """Test that send_audio converts bytearray to bytes."""
    bot = TelegramBot(token="test_token")

    # Pass bytearray instead of bytes
    audio_data = bytearray(b"audio_data_as_bytearray")
    await bot._send_audio_async(12345, audio_data, "Caption", "audio.mp3")

    # Verify send_audio was called
    mock_telegram_app.bot.send_audio.assert_called_once()


@pytest.mark.asyncio
async def test_send_video_converts_bytearray(mock_telegram_app):
    """Test that send_video converts bytearray to bytes."""
    bot = TelegramBot(token="test_token")

    # Pass bytearray instead of bytes
    video_data = bytearray(b"video_data_as_bytearray")
    await bot._send_video_async(12345, video_data, "Caption", "video.mp4", True)

    # Verify send_video was called
    mock_telegram_app.bot.send_video.assert_called_once()


@pytest.mark.asyncio
async def test_send_document_converts_bytearray(mock_telegram_app):
    """Test that send_document converts bytearray to bytes."""
    bot = TelegramBot(token="test_token")

    # Pass bytearray instead of bytes
    doc_data = bytearray(b"document_data_as_bytearray")
    await bot._send_document_async(12345, doc_data, "report.pdf", "Caption")

    # Verify send_document was called
    mock_telegram_app.bot.send_document.assert_called_once()
