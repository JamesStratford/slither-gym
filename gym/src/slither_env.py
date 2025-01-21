import numpy as np
from gymnasium import Env, spaces
import queue
import matplotlib.pyplot as plt
import pygame


class GameConnection:
    def __init__(self):
        self.latest_state = None
        self.latest_action = None
        self.queue = queue.Queue()
        self.rollout_state = False

    def put_state(self, state):
        if not self.rollout_state:
            self.latest_state = state
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
        self.step_counter = 0  # Initialize a step counter

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10, grid_size, grid_size), dtype=np.float32
        )
        # (xt, yt, accelerate)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.window_size = (800, 600)  # Set the window size
        self.screen = None  # Pygame screen
        self.clock = pygame.time.Clock()

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
        self.step_counter += 1  # Increment the step counter
        # self.render(obs)
        return obs, reward, done, False, info

    def _wait_for_next_state(self):
        # Just block on the synchronous queue
        return self.connection.get_state()

    def encode_state(self, state):
        """
        Encodes the game state into a grid-based representation.
        """
        slither = state["slither"]
        target_slither = state["target_slither"]
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
        state_grid = np.zeros(
            (10, self.grid_size, self.grid_size), dtype=np.float32)

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
            gx, gy = to_grid_coords(
                food["x"], food["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[0, gx, gy] += food.get("value", 1.0)

        # -------------------------------------------------------------------------
        # 2) OWN SLITHER: place each part of your snake in channel 1
        # -------------------------------------------------------------------------
        head_x, head_y = to_grid_coords(
            slither["x"], slither["y"], slither["x"], slither["y"])
        state_grid[1, head_x, head_y] = slither.get("ang", 0.0)
        for part in slither["parts"]:
            gx, gy = to_grid_coords(
                part["x"], part["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[2, gx, gy] = part.get("size", 10.0)

        # -------------------------------------------------------------------------
        # 3) TOP 100 BODY PARTS: place into channel 2
        # -------------------------------------------------------------------------
        for part in top_body_parts:
            gx, gy = to_grid_coords(
                part["x"], part["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[3, gx, gy] = part.get("size", 10.0)

        # -------------------------------------------------------------------------
        # 4) TARGET SLITHER: place into channel 3
        # -------------------------------------------------------------------------
        if target_slither:
            for part in target_slither.get("parts", []):
                gx, gy = to_grid_coords(
                    part.get("x", 0.0), part.get("y", 0.0), slither["x"], slither["y"]
                )
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    state_grid[4, gx, gy] = part.get("size", 10.0)
            target_x = target_slither.get("x", None)
            target_y = target_slither.get("y", None)
            if target_x is not None and target_y is not None:
                target_head_x, target_head_y = to_grid_coords(
                    target_x, target_y, slither["x"], slither["y"]
                )
                state_grid[5, target_head_x, target_head_y] = target_slither.get("ang", 0.0)
            else:
                state_grid[5, :, :] = 0.0
        else:
            state_grid[4, :, :] = 0.0
            state_grid[5, :, :] = 0.0

        # -------------------------------------------------------------------------
        # 5) PREYS: place into channel 3
        # -------------------------------------------------------------------------
        for prey in preys:
            gx, gy = to_grid_coords(
                prey["x"], prey["y"], slither["x"], slither["y"])
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                state_grid[6, gx, gy] = 1.0

        # -------------------------------------------------------------------------
        # 6) OTHER SLITHERS: body presence in channel 4, size in channel 5
        # -------------------------------------------------------------------------
        for other_snake in others:
            o_head_x, o_head_y = to_grid_coords(
                other_snake["x"], other_snake["y"], slither["x"], slither["y"])
            state_grid[7, o_head_x, o_head_y] = other_snake.get("ang", 0.0)
            state_grid[8, o_head_x, o_head_y] = other_snake.get("size", 0.0)
            for part in other_snake["parts"]:
                gx, gy = to_grid_coords(
                    part["x"], part["y"], slither["x"], slither["y"])
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    state_grid[9, gx, gy] = part.get("size", 10.0)

        return state_grid

    def calc_reward(self, payload):
        """
        Reward function logic.
        """
        reward = 0.0
        reward += max(-1, payload["slither"]["food_eaten"]
                      ) * (payload["slither"]["size"] * 0.05)
        # If any other slithers died, reward the agent
        if any(other_snake.get("dead", False) for other_snake in payload["others"]):
            reward += 100.0
        # If the agent is dead, penalize it
        if payload.get("dead", False):
            reward -= 100.0

        # Check if any other slither's head is within 20 units of your slither's body parts
        for other_snake in payload["others"]:
            if other_snake.get("dead", False):
                head_pos = np.array(
                    [other_snake["x"], other_snake["y"]])
                for part in payload["slither"]["parts"]:
                    part_pos = np.array([part["x"], part["y"]])
                    distance = np.linalg.norm(head_pos - part_pos)
                    if distance <= 20:
                        reward += 100.0
                        break  # Exit the loop once a reward is given
        return reward

    def render(self, obs):
        """
        Render the grid for visualization using Pygame.
        """
        if not obs.shape == (10, self.grid_size, self.grid_size):
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption('Slither Environment')

        # Clear the screen
        self.screen.fill((0, 0, 0))  # Black background

        # Define colors for each channel
        colors = [
            (255, 0, 0),    # Food
            (255, 255, 255),  # Slither Head
            (0, 255, 0),    # Own Slither
            (0, 0, 255),    # Top Body Parts
            (255, 255, 0),  # Preys
            (255, 255, 255),  # Other Slithers' Head
            (255, 0, 255),  # Other Slithers' Body
            (0, 255, 255),  # Other Slithers' Size
        ]

        # Render each channel
        for i in range(len(colors)):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    value = obs[i, x, y]
                    if value > 0:
                        if i == 0:
                            color = (min(255, (value * 3) + 155), 0, 0)
                        else:
                            color = colors[i]
                        # Scale grid coordinates to window size
                        rect = pygame.Rect(
                            x * (self.window_size[0] // self.grid_size),
                            y * (self.window_size[1] // self.grid_size),
                            self.window_size[0] // self.grid_size,
                            self.window_size[1] // self.grid_size
                        )
                        pygame.draw.rect(self.screen, color, rect)
        # Text
        font = pygame.font.Font(None, 20)
        # Retrieve slither angle and size
        slither_angle = self.connection.latest_state["slither"].get("ang", 0.0)
        slither_size = self.connection.latest_state["slither"].get("size", 0.0)
        # Render text with step counter, slither angle, and size
        text = font.render(
            f"Step: {self.step_counter} | Angle: {
                slither_angle:.2f} | Size: {slither_size:.2f}",
            True, (255, 255, 255)
        )
        self.screen.blit(text, (10, 10))

        pygame.display.flip()  # Update the display
        self.clock.tick(60)  # Limit to 60 FPS

    def close(self):
        """
        Close the environment and Pygame.
        """
        super().close()
        if self.screen is not None:
            pygame.quit()
