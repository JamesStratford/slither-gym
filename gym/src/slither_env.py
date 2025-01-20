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

    def __init__(self, connection: GameConnection, grid_size=50, view_range=2000):
        super().__init__()
        self.connection = connection
        self.grid_size = grid_size
        self.view_range = view_range

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7, grid_size, grid_size), dtype=np.float32
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
        top_body_parts = state["top_body_parts"]
        preys = state["preys"]

        # Suppose we use 7 channels in total:
        # 0: Food (with value)
        # 1: Own slither body parts
        # 2: Top 100 body parts (from all snakes)
        # 3: Prey locations
        # 4: Other slithers' body presence
        # 5: Other slithers' size
        # 6: Own slither angle
        state_grid = np.zeros((7, self.grid_size, self.grid_size), dtype=np.float32)

        # Helper function to map game coordinates to grid coordinates
        def to_grid_coords(x, y, agent_x, agent_y):
            rel_x = x - agent_x
            rel_y = y - agent_y
            scale = self.grid_size / (2.0 * self.view_range)
            grid_x = int((rel_x + self.view_range) * scale)
            grid_y = int((rel_y + self.view_range) * scale)
            return grid_x, grid_y

        # -------------------------------------------------------------------------
        # 1) FOOD: place food (with "value") into channel 0
        # -------------------------------------------------------------------------
        for food in foods:
            gx, gy = to_grid_coords(food["x"], food["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[0, gx, gy] += food.get("value", 1.0)

        # -------------------------------------------------------------------------
        # 2) OWN SLITHER: place each part of your snake in channel 1
        # -------------------------------------------------------------------------
        for part in slither["parts"]:
            gx, gy = to_grid_coords(part["x"], part["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[1, gx, gy] = 1.0

        # -------------------------------------------------------------------------
        # 3) TOP 100 BODY PARTS: place into channel 2
        # -------------------------------------------------------------------------
        for part in top_body_parts:
            gx, gy = to_grid_coords(part["x"], part["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[2, gx, gy] = 1.0

        # -------------------------------------------------------------------------
        # 4) PREYS: place into channel 3
        # -------------------------------------------------------------------------
        for prey in preys:
            gx, gy = to_grid_coords(prey["x"], prey["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[3, gx, gy] = 1.0

        # -------------------------------------------------------------------------
        # 5) OTHER SLITHERS: body presence in channel 4, size in channel 5
        # -------------------------------------------------------------------------
        for other_snake in others:
            size = other_snake.get("size", 1.0)
            for part in other_snake["parts"]:
                gx, gy = to_grid_coords(part["x"], part["y"], slither["x"], slither["y"])
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    state_grid[4, gx, gy] = 1.0  # Body presence
                    state_grid[5, gx, gy] += size  # Snake size

        # -------------------------------------------------------------------------
        # 6) OWN SLITHER ANGLE in channel 6 (fill entire channel)
        # -------------------------------------------------------------------------
        slither_angle = slither.get("ang", 0.0)
        state_grid[6, :, :] = slither_angle

        return state_grid



    def calc_reward(self, payload):
        """
        Reward function logic.
        """
        reward = 0.0
        reward += payload["slither"]["food_eaten"]
        # If any other slithers died, reward the agent
        if any(other_snake.get("dead", False) for other_snake in payload["others"]):
            reward += 100.0
        # If the agent is dead, penalize it
        if payload.get("dead", False):
            reward -= 100.0
        # Check if any other slither's head is within 20 units of your slither's body parts
        for other_snake in payload["others"]:
            if other_snake.get("dead", False):
                head_pos = np.array([other_snake["parts"][0]["x"], other_snake["parts"][0]["y"]])
                for part in payload["slither"]["parts"]:
                    part_pos = np.array([part["x"], part["y"]])
                    distance = np.linalg.norm(head_pos - part_pos)
                    if distance <= 20:
                        reward += 100.0
                        break  # Exit the loop once a reward is given
        if reward != 0.0:
            print(reward)
        return reward

    def close(self):
        """
        Close the environment.
        """
        super().close()
