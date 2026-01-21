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
    assert bot.photo_timeout == 60
    assert bot.max_retries == 3
    assert bot.auto_set_commands is True


def test_telegram_bot_init_with_custom_settings(mock_telegram_app):
    """Test bot initialization with custom timeout and retry settings."""
    bot = TelegramBot(
        token="test_token",
        message_timeout=45,
        photo_timeout=90,
        max_retries=5,
        auto_set_commands=False,
    )
    assert bot.message_timeout == 45
    assert bot.photo_timeout == 90
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
async def test_create_message_handler_extracts_message_data():
    """Test that _create_message_handler properly extracts data into Message object."""
    bot = TelegramBot(token="test_token")

    received_message = None

    def handler(msg: Message):
        nonlocal received_message
        received_message = msg

    # Create mock Update
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    mock_update.message.text = "Hello World"
    mock_update.message.photo = None
    mock_update.effective_user.id = 67890
    mock_update.effective_user.first_name = "John"
    mock_update.effective_user.username = "johndoe"

    # Create wrapped handler
    wrapped = bot._create_message_handler(handler)

    # Execute handler
    await wrapped(mock_update, MagicMock())

    # Verify Message object was created correctly
    assert received_message is not None
    assert received_message.chat_id == 12345
    assert received_message.text == "Hello World"
    assert received_message.user_id == 67890
    assert received_message.user_name == "John"
    assert received_message.username == "johndoe"
    assert received_message.photo_bytes is None


@pytest.mark.asyncio
async def test_create_message_handler_downloads_photo():
    """Test that _create_message_handler downloads photos automatically."""
    bot = TelegramBot(token="test_token")

    received_message = None

    def handler(msg: Message):
        nonlocal received_message
        received_message = msg

    # Create mock Update with photo
    mock_update = MagicMock()
    mock_update.message.chat_id = 12345
    mock_update.message.text = None
    mock_update.effective_user.id = 67890
    mock_update.effective_user.first_name = "Jane"
    mock_update.effective_user.username = "janedoe"

    # Mock photo download
    mock_photo = MagicMock()
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=b"photo_data_123")
    mock_photo.get_file = AsyncMock(return_value=mock_file)
    mock_update.message.photo = [mock_photo]

    # Create wrapped handler
    wrapped = bot._create_message_handler(handler)

    # Execute handler
    await wrapped(mock_update, MagicMock())

    # Verify photo was downloaded
    assert received_message is not None
    assert received_message.photo_bytes == b"photo_data_123"
    assert received_message.text is None


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
    mock_telegram_app.bot.send_photo.assert_called_once_with(
        chat_id=12345, photo=b"photo_data", caption="Test caption", read_timeout=60, write_timeout=60
    )


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
    """Test that send() returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send(12345, "Test message")
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


def test_schedule_message_aliases(mock_telegram_app):
    """Test that schedule() is an alias for schedule_message()."""
    bot = TelegramBot(token="test_token")
    # Both methods should exist
    assert hasattr(bot, "schedule")
    assert hasattr(bot, "schedule_message")


def test_cancel_schedule_aliases(mock_telegram_app):
    """Test that cancel_schedule() is an alias for cancel_scheduled_message()."""
    bot = TelegramBot(token="test_token")
    # Both methods should exist
    assert hasattr(bot, "cancel_schedule")
    assert hasattr(bot, "cancel_scheduled_message")


def test_message_dataclass_creation():
    """Test that Message dataclass can be created with required fields."""
    msg = Message(chat_id=12345)
    assert msg.chat_id == 12345
    assert msg.text is None
    assert msg.user_id is None
    assert msg.user_name is None
    assert msg.username is None
    assert msg.photo_bytes is None


def test_message_dataclass_with_all_fields():
    """Test Message dataclass with all fields populated."""
    msg = Message(
        chat_id=12345,
        text="Hello",
        user_id=67890,
        user_name="John",
        username="johndoe",
        photo_bytes=b"photo_data",
    )
    assert msg.chat_id == 12345
    assert msg.text == "Hello"
    assert msg.user_id == 67890
    assert msg.user_name == "John"
    assert msg.username == "johndoe"
    assert msg.photo_bytes == b"photo_data"


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
