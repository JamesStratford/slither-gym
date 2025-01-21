import asyncio
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from slither_env import SlitherEnv, GameConnection

async def render(connection: GameConnection, env: SlitherEnv):
    last_state = None
    try:
        while True:
            state = connection.latest_state
            if last_state != state:
                last_state = state
                obs = env.encode_state(state)
                env.render(obs)
            # Add a small delay to yield control to the event loop
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in render loop: {e}")
