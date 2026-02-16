# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

# EXAMPLE_NAME = "Bot with user ID whitelist"
# EXAMPLE_REQUIRES = "Requires TELEGRAM_BOT_TOKEN environment variable and authorized user IDs."

from arduino.app_bricks.telegram_bot import TelegramBot, Sender, Message
from arduino.app_utils import App

# Replace with your authorized Telegram user IDs
# Use @userinfobot on Telegram to get your user ID
AUTHORIZED_USER_IDS = [123456789, 987654321]

bot = TelegramBot(whitelist_user_ids=AUTHORIZED_USER_IDS)


def restricted_command(sender: Sender, message: Message):
    """Only authorized users can trigger this."""
    sender.reply(f"✅ Access granted! Welcome {sender.first_name}.")


bot.add_command("start", restricted_command, "Start the bot (authorized users only)")

App.run()
