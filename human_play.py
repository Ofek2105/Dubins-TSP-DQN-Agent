import arcade
import math
from RLenviroment import TSPPlaneEnv  # replace with your actual env filename
import time

WINDOW_SIZE = 600
TURN_ANGLE = 15  # degrees
SPACESHIP_SPEED = 0.01

class HumanPlay(arcade.Window):
    def __init__(self, env):
        super().__init__(WINDOW_SIZE, WINDOW_SIZE, "TSP Plane - Human Play")
        self.env = env
        arcade.set_background_color(arcade.color.BLACK)
        self.action_map = {
            arcade.key.LEFT: 0,   # turn left
            arcade.key.RIGHT: 1,  # turn right
            arcade.key.UP: 2      # go straight
        }
        self.keys_held = set()

    def on_draw(self):
        arcade.start_render()
        for i, city in enumerate(self.env.cities):
            x = city[0] * WINDOW_SIZE
            y = city[1] * WINDOW_SIZE
            if self.env.visited[i]:
                arcade.draw_circle_filled(x, y, 6, arcade.color.GREEN)  # visited city in green
            else:
                arcade.draw_circle_filled(x, y, 6, arcade.color.RED)  # unvisited city in red

        x = self.env.agent_position[0] * WINDOW_SIZE
        y = self.env.agent_position[1] * WINDOW_SIZE
        arcade.draw_circle_filled(x, y, 10, arcade.color.BLUE)

        angle_rad = math.radians(self.env.agent_angle)
        dx = 20 * math.cos(angle_rad)
        dy = 20 * math.sin(angle_rad)
        arcade.draw_line(x, y, x + dx, y + dy, arcade.color.WHITE, 2)

    def on_key_press(self, key, modifiers):
        if key in self.action_map:
            self.keys_held.add(key)

    def on_key_release(self, key, modifiers):
        if key in self.action_map and key in self.keys_held:
            self.keys_held.remove(key)

    def on_update(self, delta_time):
        if not self.keys_held:
            action = 2  # go straight by default
        else:
            # pick one action based on pressed keys: left, right, straight
            # Prioritize left, then right, then straight if multiple keys held
            if arcade.key.LEFT in self.keys_held:
                action = 0
            elif arcade.key.RIGHT in self.keys_held:
                action = 1
            else:
                action = 2

        obs, reward, done, truncated, info = self.env.step(action)
        print(self.env.step_count)
        if done:
            print("Episode finished!")
            self.env.reset()

def main():
    env = TSPPlaneEnv(num_cities=5, frame_skip=0, verbose=True, max_steps=500, speed=0.008, angle_turn=20)
    env.reset()
    window = HumanPlay(env)
    arcade.run()

if __name__ == "__main__":
    main()
