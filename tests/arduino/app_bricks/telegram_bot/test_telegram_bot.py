# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

import pytest
import inspect
from unittest.mock import MagicMock, AsyncMock, patch
from arduino.app_bricks.telegram_bot import TelegramBot
from telegram.error import NetworkError, TimedOut


@pytest.fixture
def mock_telegram_app(monkeypatch):
    """Mock the Telegram Application to avoid real network calls."""
    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.bot.send_message = AsyncMock()
    mock_app.bot.send_photo = AsyncMock()
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


def test_telegram_bot_init_with_custom_settings(mock_telegram_app):
    """Test bot initialization with custom timeout and retry settings."""
    bot = TelegramBot(token="test_token", message_timeout=45, photo_timeout=90, max_retries=5)
    assert bot.message_timeout == 45
    assert bot.photo_timeout == 90
    assert bot.max_retries == 5


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


def test_add_command_sync_callback(mock_telegram_app):
    """Test registering a synchronous command handler."""
    bot = TelegramBot(token="test_token")

    def sync_handler(update, context):
        pass

    bot.add_command("start", sync_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_add_command_async_callback(mock_telegram_app):
    """Test registering an async command handler."""
    bot = TelegramBot(token="test_token")

    async def async_handler(update, context):
        pass

    bot.add_command("help", async_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_on_text_handler(mock_telegram_app):
    """Test registering a text message handler."""
    bot = TelegramBot(token="test_token")

    def text_handler(update, context):
        pass

    bot.on_text(text_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_on_photo_handler(mock_telegram_app):
    """Test registering a photo message handler."""
    bot = TelegramBot(token="test_token")

    def photo_handler(update, context):
        pass

    bot.on_photo(photo_handler)
    mock_telegram_app.add_handler.assert_called_once()


def test_make_async_handler_with_sync_function(mock_telegram_app):
    """Test that synchronous callbacks are wrapped correctly."""
    bot = TelegramBot(token="test_token")

    def sync_callback(update, context):
        return "sync_result"

    wrapped = bot._make_async_handler(sync_callback)
    assert inspect.iscoroutinefunction(wrapped)


def test_make_async_handler_with_async_function(mock_telegram_app):
    """Test that async callbacks are not double-wrapped."""
    bot = TelegramBot(token="test_token")

    async def async_callback(update, context):
        return "async_result"

    wrapped = bot._make_async_handler(async_callback)
    assert wrapped is async_callback


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
    await bot._send_photo_async(12345, photo="photo.jpg", caption="Test caption")
    mock_telegram_app.bot.send_photo.assert_called_once_with(
        chat_id=12345, photo="photo.jpg", caption="Test caption", read_timeout=60, write_timeout=60
    )


@pytest.mark.asyncio
async def test_send_photo_async_timeout(mock_telegram_app):
    """Test that timeout errors are properly raised."""
    bot = TelegramBot(token="test_token")
    mock_telegram_app.bot.send_photo.side_effect = TimedOut("Request timed out")

    with pytest.raises(TimedOut):
        await bot._send_photo_async(12345, "photo.jpg")


@pytest.mark.asyncio
async def test_get_photo_async(mock_telegram_app):
    """Test internal async get_photo method."""
    bot = TelegramBot(token="test_token")

    # Mock update with photo
    mock_update = MagicMock()
    mock_photo = MagicMock()
    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=b"photo_bytes")
    mock_photo.get_file = AsyncMock(return_value=mock_file)
    mock_update.message.photo = [mock_photo]

    mock_context = MagicMock()

    photo_bytes = await bot._get_photo_async(mock_update, mock_context)
    assert photo_bytes == b"photo_bytes"


def test_send_message_bot_not_initialized(mock_telegram_app):
    """Test that send_message returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_message(12345, "Test message")
    assert result is False


def test_send_photo_bot_not_initialized(mock_telegram_app):
    """Test that send_photo returns False when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    result = bot.send_photo(12345, "photo.jpg")
    assert result is False


def test_get_photo_bot_not_initialized(mock_telegram_app):
    """Test that get_photo returns None when bot is not initialized."""
    bot = TelegramBot(token="test_token")
    mock_update = MagicMock()
    mock_context = MagicMock()
    result = bot.get_photo(mock_update, mock_context)
    assert result is None


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
    """Test that bot is not running after initialization."""
    bot = TelegramBot(token="test_token")
    assert bot._running is False


def test_stop_when_not_running(mock_telegram_app):
    """Test that stop() does nothing when bot is not running."""
    bot = TelegramBot(token="test_token")
    bot.stop()  # Should not raise
    assert bot._running is False
