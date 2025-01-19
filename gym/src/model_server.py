import asyncio
import websockets
import json
from stable_baselines3 import PPO
from slither_env import SlitherEnv, GameConnection
from utils import handle_client, setup_environment

connection = GameConnection()

async def main():
    setup_environment(connection)

    server = await websockets.serve(lambda ws: handle_client(ws, connection), "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())