import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import time

from NagelSchreckenbergSingle import NagelSchreckenbergSingle

class SimulationSingle:
    def __init__(self, road_length=100, lag_parameter=0.3, entry_rate=0.5):
        self.road_length = road_length
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate

        self.model = NagelSchreckenbergSingle(self.road_length, 0, 5, self.lag_parameter, self.entry_rate)
        self.data = np.array([])
        self.steps = 0

    def run(self, warmup_steps=0, steps=0, warmup_steps_mult=5, steps_mult=10):
        ws = warmup_steps_mult * self.road_length
        if warmup_steps:
            ws = warmup_steps

        s = ws * steps_mult
        if steps:
            s = steps
        self.steps = s

        for _ in range(ws):
            self.model.step()

        data = []
        for _ in range(s):
            data.append(self.model.get_road())
            self.model.step()
        self.data = np.array(data)

    def reset(self):
        self.model = NagelSchreckenbergSingle(self.road_length, 0, 5, self.lag_parameter, self.entry_rate)
        self.data = np.array([])
        self.steps = 0

    def plot(self, figsize=(12, 6), single_color=False):
        fig, ax = plt.subplots(figsize=figsize)

        colors = ["white", "#8B0000", "red", "orange", "yellow", "lightgreen", "green"]
        cmap = ListedColormap(colors)
        bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        norm = BoundaryNorm(bounds, cmap.N)

        if single_color:
            colors = ["white", "black"]
            cmap = ListedColormap(colors)
            bounds = [-1.5, -0.5, 5.5]
            norm = BoundaryNorm(bounds, cmap.N)

        img = ax.imshow(self.data, cmap=cmap, norm=norm, interpolation="none", aspect="auto")
        ax.set_xlabel("Road Position")
        ax.set_ylabel("Time Step")
        ax.set_title("Nagel-Schreckenberg Traffic Simulation")
        plt.show()

    def animate(self, interval=100, figsize=(10, 2), single_color=False):
        fig, ax = plt.subplots(figsize=figsize)

        road_data = self.data[0]
        x = np.where(road_data == 1)[0]
        y = np.ones_like(x)

        ax.bar(x, y, width=1.0, color="black")
        self.animate_help_meta(ax, 0)

        color_dict = { 0: "#8B0000", 1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green" }
        if single_color:
            color_dict = {}
        map_color = np.vectorize(lambda x: color_dict.get(x, "black"))

        def update(frame):
            road_data = self.data[frame]

            x = np.where(road_data >= 0)[0]
            y = np.ones_like(x)

            ax.clear()
            ax.bar(x, y, width=1.0, color=map_color(road_data[x]))
            self.animate_help_meta(ax, frame)
            return ax.patches

        ani = animation.FuncAnimation(fig, update, frames=self.steps, interval=interval)
        plt.show()

    def animate_help_meta(self, ax, frame):
        ax.set_xlim(-0.5, self.road_length-0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title("Nagel-Schreckenberg Traffic Flow (%s)" % frame)
        ax.set_xlabel("Road Position")

    def get_stats(self):
        time_stats = self.model.get_time_stats()
        return {
            "length": self.road_length * 7.5,
            "length_model": self.road_length,
            "time_min": np.min(time_stats),
            "time_max": np.max(time_stats),
            "time_avg": np.average(time_stats),
            "velocity_avg": self.road_length * 7.5 / np.average(time_stats)
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"Total road segment length: {stats["length"]:.0f}m")
        print(f"Average completion time: {stats["time_avg"]:.2f}s")
        print(f"Average speed: {stats["velocity_avg"]:.2f}m/s or {stats["velocity_avg"]*3.6:.0f}km/h")

if __name__ == "__main__":
    start = time.time()
    sim = SimulationSingle(500, 0.3, 0.3)
    sim.run()
    sim.print_stats()
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} seconds")