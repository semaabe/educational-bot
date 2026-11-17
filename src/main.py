import asyncio, uvicorn, sys, os, discord, subprocess
from discord.ext import commands
from community_apps.getMessageDiscord import DiscordBot
from backend.app import app as fastapi_app
from router.RouteMap import ThreadSafeMap
from community_apps.discordHelper import get_from_app
from utlis.config import require_discord_token
from backend.auth import auth_headers
import httpx
import logging
from router.utterances import UTTERANCES

# temp
from databases.chroma.crudChroma import CRUD

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define bot and command prefix
'''could use Intents.all() instead of Intents.default()'''
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize your Discord bot class 
discord_bot = DiscordBot(bot)

# Function to run FastAPI server
async def run_fastapi():
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

# Function to run the Discord bot
async def run_discord_bot():
    await bot.start(require_discord_token())

async def wait_for_backend(timeout=10):
    start = asyncio.get_event_loop().time()
    while True:
        try:
            resp = await get_from_app("health")
            if resp.status_code == 200:
                return
        except Exception as e:
            logging.error(f'Backend Startup Error: {e}')

        if asyncio.get_event_loop().time() - start > timeout:
            raise RuntimeError("Backend never became ready")

        await asyncio.sleep(0.5)


# Register newly joined Discord members through the same backend endpoint used
# by the startup synchronization in DiscordBot.
@bot.event
async def on_member_join(member):
    if member.bot:
        return
    payload = {
        "username": str(member.id),
        "email": f"{member.id}@discord.local",
        "role": "student",
        "default_collection": "general",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/api/users", json=payload, headers=auth_headers()
            )
        if response.status_code not in (200, 201):
            logging.error("Failed to register Discord member %s: %s", member.id, response.status_code)
    except httpx.HTTPError as exc:
        logging.error("Failed to register Discord member %s: %s", member.id, exc)

# Main function to run both FastAPI and Discord bot concurrently
async def main():
    routes = ThreadSafeMap()

    fastapi_task = asyncio.create_task(run_fastapi())

    await wait_for_backend()

    response = await get_from_app("collections")
    if not response or response.status_code != 200:
        raise RuntimeError("Failed to load collections from backend")

    collections_data = response.json()

    if isinstance(collections_data, dict):
        collections = collections_data.get("collections", [])
    else:
        collections = collections_data

    if not isinstance(collections, list):
        raise RuntimeError(
            f"Unexpected collections response type: {type(collections_data).__name__}"
        )
    for collection in collections:
        name = collection.get("name")
        description = collection.get("description")

        if not name or not description:
            continue
        utterances = await UTTERANCES.get(name) or []
        await routes.set(name, utterances)

    # Send the routes map to backend to set up semantic router
    async with httpx.AsyncClient() as client:
        routes_snapshot = await routes.snapshot()
        await client.post(
            "http://localhost:8000/setup_routes",
            json=routes_snapshot,
            headers=auth_headers(),
        )

    discord_bot.set_routes(routes)

    discord_task = asyncio.create_task(run_discord_bot())

    # Run both tasks concurrently
    await asyncio.gather(fastapi_task, discord_task)

    # Main instance of the map

    

# Entry point
if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("Shutting down...")


# async def load_pdfs():
#     # print current working directory
#     crud = CRUD()
#     await crud.save_pdfs("./src/services/pdf_files", "course_materials")


# if __name__ == "__main__":
#     asyncio.run(load_pdfs())


'''
this, along with the !update function, should be moved to a separate file
responsible for model initialization
'''
