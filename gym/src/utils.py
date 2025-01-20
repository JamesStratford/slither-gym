import asyncio
import json
import websockets
from stable_baselines3 import PPO, DDPG
from slither_env import SlitherEnv, GameConnection

async def handle_client(websocket, connection: GameConnection):
    print(f"Client connected from {websocket.remote_address}")
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "update":
                payload = data["payload"]
                connection.put_state(payload)

                action = connection.latest_action
                if action is None:
                    action = [0.0, 0.0, 0.0]

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


def setup_environment(connection: GameConnection, train: bool = False, *, policy_kwargs=None):
    env = SlitherEnv(connection=connection)
    if train:
        model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
        return env, model
    else:
        env.model = PPO.load("slither_model", env=env, verbose=1, policy_kwargs=policy_kwargs)
        return env 