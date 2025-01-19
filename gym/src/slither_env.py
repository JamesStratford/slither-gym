import numpy as np
from gymnasium import Env, spaces
import queue


class GameConnection:
    def __init__(self):
        self.latest_state = None
        self.latest_action = None
        self.queue = queue.Queue()

    def put_state(self, state):
        self.queue.put(state)

    def get_state(self):
        return self.queue.get()

    def set_action(self, action):
        self.latest_action = action


class SlitherEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, connection: GameConnection, grid_size=20, view_range=1000):
        super().__init__()
        self.connection = connection
        self.grid_size = grid_size
        self.view_range = view_range

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, grid_size, grid_size), dtype=np.float32
        )
        # (xt, yt, accelerate)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Wait for the next incoming state from the queue
        state = self._wait_for_next_state()
        obs = self.encode_state(state)
        info = {}
        return obs, info

    def step(self, action):
        # Store the chosen action so the server can send it to the JS client
        self.connection.set_action(action)
        # Wait for the next incoming game state
        next_state = self._wait_for_next_state()

        # Convert to observation, compute reward, check done
        obs = self.encode_state(next_state)
        reward = self.calc_reward(next_state)
        done = bool(next_state.get("dead", False))

        info = {}
        return obs, reward, done, False, info

    def _wait_for_next_state(self):
        # Just block on the synchronous queue
        return self.connection.get_state()

    def encode_state(self, state):
        """
        Encodes the game state into a grid-based representation.
        """
        slither = state["slither"]
        foods = state["foods"]
        others = state["others"]

        state_grid = np.zeros(
            (5, self.grid_size, self.grid_size), dtype=np.float32)
        agent_position = (slither["x"], slither["y"])
        agent_direction = (slither["xm"], slither["ym"])

        def add_to_grid(entities, channel, default_val=1.0):
            for entity in entities:
                rel_x = entity["x"] - agent_position[0]
                rel_y = entity["y"] - agent_position[1]
                scale = self.grid_size / (2.0 * self.view_range)
                grid_x = int((rel_x + self.view_range) * scale)
                grid_y = int((rel_y + self.view_range) * scale)

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # If the entity has a "value" field, use that; otherwise fall back to default_val
                    val = entity.get("value", default_val)
                    state_grid[channel, grid_x, grid_y] += val

        # 0: food (add the actual "value" instead of +1)
        add_to_grid(foods, 0, default_val=1.0)

        # 1: other slither parts
        for other in others:
            add_to_grid(other["parts"], 1, default_val=1.0)

        # 2: own slither parts
        add_to_grid(slither["parts"], 2, default_val=1.0)

        # 3,4: agent direction
        state_grid[3, :, :] = agent_direction[0]
        state_grid[4, :, :] = agent_direction[1]

        return state_grid

    def calc_reward(self, payload):
        """
        Reward function logic.
        """
        reward = 0.0
        reward += payload["slither"]["size"] * 0.5
        reward += payload["slither"]["food_eaten"] * 10
        if payload.get("dead", False):
            reward -= 100.0
        return reward

    def close(self):
        """
        Close the environment.
        """
        super().close()
