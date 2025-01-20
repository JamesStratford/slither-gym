from slither_env import GameConnection
import websockets
import asyncio
from model import ViTExtractor
from utils import handle_client
from stable_baselines3 import PPO
from slither_env import SlitherEnv

GRID_SIZE = 50
policy_kwargs = dict(
    features_extractor_class=ViTExtractor,
    features_extractor_kwargs=dict(num_heads=4)
)


connection = GameConnection()


async def main():
    env = SlitherEnv(connection=connection, grid_size=GRID_SIZE)
    model = PPO("CnnPolicy", env, verbose=1,
                policy_kwargs=policy_kwargs, n_steps=1024, batch_size=64)
    await asyncio.to_thread(model.learn, total_timesteps=50000)
    model.save("slither_model")
    env.close()


async def start_server():
    server = await websockets.serve(
        lambda ws: handle_client(ws, connection),
        "127.0.0.1",
        8881,
        ping_interval=20,
        ping_timeout=10
    )
    print("WebSocket server started at ws://127.0.0.1:8765")
    await server.wait_closed()


async def main_async():
    await asyncio.gather(
        start_server(),
        main()  # Ensure this is awaited
    )

if __name__ == "__main__":
    asyncio.run(main_async())
