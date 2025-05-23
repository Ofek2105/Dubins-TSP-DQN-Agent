import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class TSPPlaneEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, num_cities=5, frame_skip=0):
        super().__init__()

        self.num_cities = num_cities
        self.frame_skip = frame_skip
        self.TURN_ANGLE = 10  # degrees
        self.SPACESHIP_SPEED = 0.01
        self.agent_angle = None
        self.action_space = spaces.Discrete(3)  # 0 = left, 1 = right, 2 = straight

        # Observation: [angle_to_city, distance_to_city] * K + heading + x + y
        self.k_nearest = min(5, self.num_cities)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * self.k_nearest + 3,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cities = np.random.rand(self.num_cities, 2) # uniform between 0 and 1
        self.visited = np.zeros(self.num_cities, dtype=bool)

        self.agent_position = np.random.rand(2)
        self.agent_angle = np.random.uniform(0, 360)

        self.steps = 0
        return self._get_obs(), {}

    def _closest_unvisited_distance(self):
        """
        Returns the Euclidean distance from the agent's current position
        to the closest unvisited city. If all cities have been visited,
        returns 0.0.
        """
        if np.all(self.visited):
            return 0.0
        return min(
            np.linalg.norm(self.agent_position - city)
            for i, city in enumerate(self.cities)
            if not self.visited[i]
        )

    def step(self, action):
        reward = 0.0
        prev_closest_dist = self._closest_unvisited_distance()

        for _ in range(self.frame_skip + 1):
            if action == 0:
                self.agent_angle += self.TURN_ANGLE
            elif action == 1:
                self.agent_angle -= self.TURN_ANGLE
            self.agent_angle %= 360

            rad = math.radians(self.agent_angle)
            dx = self.SPACESHIP_SPEED * math.cos(rad)
            dy = self.SPACESHIP_SPEED * math.sin(rad)
            self.agent_position += np.array([dx, dy])
            self.agent_position = np.clip(self.agent_position, 0, 1)

            for i, city in enumerate(self.cities):
                if not self.visited[i] and np.linalg.norm(self.agent_position - city) < 0.03:
                    self.visited[i] = True
                    reward += 10.0

        new_closest_dist = self._closest_unvisited_distance()
        reward += (prev_closest_dist - new_closest_dist) * 2.0

        if np.all(self.visited):
            done = True
            reward += 20.0
        else:
            done = False
            reward += -0.01 * (self.frame_skip + 1)
            reward += 1.0 * np.sum(self.visited)

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        unvisited = self.cities[~self.visited]
        if len(unvisited) == 0:
            padded = np.zeros((self.k_nearest, 2))
        else:
            dists = np.linalg.norm(unvisited - self.agent_position, axis=1)
            idxs = np.argsort(dists)[:self.k_nearest]
            nearest = unvisited[idxs]
            rel = nearest - self.agent_position

            angles_norm = self._get_rel_norm_deg(rel)
            dist_norm = 2 * np.linalg.norm(rel, axis=1) / np.sqrt(2) - 1

            padded = np.zeros((self.k_nearest, 2))
            padded[:len(nearest), 0] = angles_norm[:len(nearest)]
            padded[:len(nearest), 1] = dist_norm[:len(nearest)]

        obs = np.concatenate([
            padded.flatten(),
            [self.agent_angle / 180.0 - 1.0],
            self.agent_position
        ]).astype(np.float32)
        return obs

    def _get_rel_norm_deg(self, rel):
        """
        gets an array of 3d vectors.
        and returns the normalized angle in degrees. between -1 and 1.
        0 correspond to 0 deg,
        1 and -1 corresponds to 180 and -180 deg,
        :param rel:
        :return:
        """
        angles = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))  # in [-180, 180]
        # Relative angle to agent's current heading, normalized to [-180, 180]
        rel_angles = (angles - self.agent_angle + 180) % 360 - 180

        return rel_angles / 180.0 # Normalize to [-1, 1]

    def render(self):
        pass  # optional

    def close(self):
        pass
