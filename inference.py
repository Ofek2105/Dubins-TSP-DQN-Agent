# inference.py
import torch
import arcade
import math
from dqn_model import DQN
from RLenviroment import TSPPlaneEnv

WINDOW_SIZE = 600

class InferencePlay(arcade.Window):
    def __init__(self, model_path, env):
        super().__init__(WINDOW_SIZE, WINDOW_SIZE, "TSP Plane - Inference Play")
        self.env = env
        self.obs, _ = self.env.reset()
        self.done = False

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n
        self.model = DQN(obs_dim, act_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.steps = 0

        arcade.set_background_color(arcade.color.BLACK)

    def on_draw(self):
        arcade.start_render()

        for i, city in enumerate(self.env.cities):
            x = city[0] * WINDOW_SIZE
            y = city[1] * WINDOW_SIZE
            color = arcade.color.GREEN if self.env.visited[i] else arcade.color.RED
            arcade.draw_circle_filled(x, y, 6, color)

        x = self.env.agent_position[0] * WINDOW_SIZE
        y = self.env.agent_position[1] * WINDOW_SIZE
        arcade.draw_circle_filled(x, y, 10, arcade.color.BLUE)

        angle_rad = math.radians(self.env.agent_angle)
        dx = 20 * math.cos(angle_rad)
        dy = 20 * math.sin(angle_rad)
        arcade.draw_line(x, y, x + dx, y + dy, arcade.color.WHITE, 2)

    def on_update(self, delta_time):
        if self.done:
            self.obs, _ = self.env.reset()
            self.done = False
            return

        with torch.no_grad():
            obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
            action = self.model(obs_tensor).argmax(dim=1).item()

        self.obs, reward, self.done, truncated, info = self.env.step(action)

def main():
    model_path = "saved_models/last.pt"  # or "models/last.pt"
    env = TSPPlaneEnv(num_cities=5, frame_skip=0, verbose=True)
    env.reset()
    window = InferencePlay(model_path, env)
    arcade.run()

if __name__ == "__main__":
    main()
