import argparse
import asyncio
from slither_env import GameConnection, SlitherEnv
from websocket_server import start_server
from renderer import render
from stable_baselines3 import PPO
from websockets.asyncio.server import ServerProtocol

import json
import websockets

GRID_SIZE = 128


async def handle_client(websocket: ServerProtocol, connection: GameConnection, model: PPO, env: SlitherEnv):
    print(f"Client connected from {websocket.remote_address}")
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if data["type"] == "update":
                payload = data["payload"]
                connection.put_state(payload)
                obs = env.encode_state(connection.latest_state)
                action, _ = model.predict(obs, deterministic=False)

                accelerate = 1 if action[2] > 0.9 else 0

                action_message = {
                    "type": "update",
                    "payload": {
                        "xt": float(action[0]),
                        "yt": float(action[1]),
                        "acceleration": accelerate
                    }
                }
                await websocket.send(json.dumps(action_message))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main_async(model_path: str):
    connection = GameConnection()
    env = SlitherEnv(connection=connection, grid_size=GRID_SIZE)
    model = PPO.load(model_path, env=env, verbose=1)
    # Start the WebSocket server and handle client connections
    await asyncio.gather(
        start_server(connection, handle_client=handle_client, model=model, env=env),
        render(connection, env)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model and start the server.")
    parser.add_argument("model_path", type=str,
                        help="Path to the model file to load.")
    args = parser.parse_args()

    asyncio.run(main_async(args.model_path))
