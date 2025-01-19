from stable_baselines3 import PPO
from slither_env import SlitherEnv, GameConnection
import websockets
import asyncio
from model import SlitherCNN
from utils import handle_client, setup_environment

policy_kwargs = dict(
    features_extractor_class=SlitherCNN,
    features_extractor_kwargs=dict(features_dim=128)
)

connection = GameConnection()

def main():
    env, model = setup_environment(connection, train=True, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=50000)
    model.save("slither_model")
    env.close()

async def start_server():
    server = await websockets.serve(
        lambda ws: handle_client(ws, connection),
        "localhost",
        8765
    )
    print("WebSocket server started at ws://localhost:8765")
    await server.wait_closed()

async def main_async():
    await asyncio.gather(
        start_server(),
        asyncio.to_thread(main)
    )

if __name__ == "__main__":
    asyncio.run(main_async())
