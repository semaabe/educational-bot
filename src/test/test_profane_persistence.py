import os
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DISCORD_TOKEN", "test-discord-token")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("DB_PATH", "./test_chromadb")
os.environ.setdefault("PROFANITY_THRESHOLD", "0.7")
os.environ.setdefault("DISTANCE_THRESHOLD", "0.7")

import community_apps.getMessageDiscord as discord_module


class FakeTree:
    def command(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    async def sync(self):
        return None


class FakeBot:
    def __init__(self):
        self.user = object()
        self.tree = FakeTree()
        self.events = {}
        self.guilds = []
        self.process_commands = AsyncMock()

    def event(self, func):
        self.events[func.__name__] = func
        return func

    def command(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator


@pytest.mark.asyncio
async def test_deleted_profane_message_is_not_persisted():
    bot = FakeBot()
    discord_module.DiscordBot(bot)
    on_message = bot.events["on_message"]

    message = SimpleNamespace(
        author=SimpleNamespace(mention="@student", name="student", id=1),
        content="This is a profane message long enough to be processed",
        channel=SimpleNamespace(id=10, name="general", send=AsyncMock()),
        guild=SimpleNamespace(id=20, name="test-guild"),
        id=30,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        delete=AsyncMock(),
    )

    with patch(
        "community_apps.getMessageDiscord.profanity_checker",
        new=AsyncMock(return_value=[0.95]),
    ), patch(
        "community_apps.getMessageDiscord.message_filter",
        new=AsyncMock(return_value=True),
    ) as mock_message_filter, patch(
        "community_apps.getMessageDiscord.get_parameters",
        new=AsyncMock(),
    ) as mock_get_parameters, patch(
        "community_apps.getMessageDiscord.update_message",
        new=AsyncMock(),
    ) as mock_update_message:
        await on_message(message)

    message.delete.assert_awaited_once()
    message.channel.send.assert_awaited()
    mock_message_filter.assert_not_awaited()
    mock_get_parameters.assert_not_awaited()
    mock_update_message.assert_not_called()
    bot.process_commands.assert_not_awaited()
