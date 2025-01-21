from slither_env import GameConnection
import asyncio
from model import ViTExtractor
from websocket_server import start_server
from renderer import render
from stable_baselines3 import PPO
from slither_env import SlitherEnv
from stable_baselines3.common.callbacks import BaseCallback

GRID_SIZE = 128
PATCH_SIZE = 8


class BackpropagationCallback(BaseCallback):
    def __init__(self, connection: GameConnection, verbose=0):
        super(BackpropagationCallback, self).__init__(verbose)
        self.connection = connection

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self.connection.rollout_state = False

    def _on_rollout_end(self) -> None:
        self.connection.rollout_state = True


async def learn(connection, env):
    policy_kwargs = dict(
        features_extractor_class=ViTExtractor,
        features_extractor_kwargs=dict(num_heads=4, patch_size=PATCH_SIZE)
    )

    model = PPO("CnnPolicy", env, verbose=1,
                policy_kwargs=policy_kwargs, n_steps=512, batch_size=32, learning_rate=0.001)
    await asyncio.to_thread(model.learn, total_timesteps=50000, callback=BackpropagationCallback(connection))
    model.save("output/slither_model")
    env.close()


async def main_async():
    connection = GameConnection()
    env = SlitherEnv(connection=connection, grid_size=GRID_SIZE)
    await asyncio.gather(
        start_server(connection),
        render(connection, env),
        learn(connection, env),
    )

if __name__ == "__main__":
    asyncio.run(main_async())
