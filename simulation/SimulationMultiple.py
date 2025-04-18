import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math

from NagelSchreckenbergMultiple import NagelSchreckenbergMultiple

class SimulationMultiple:
    def __init__(self, road_length=100, lanes=2, lag_parameter=0.3, entry_rate=0.5, multi_lane_rules=True):
        self.road_length = road_length
        self.lanes = lanes
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate
        self.multi_lane_rules = multi_lane_rules

        self.model = NagelSchreckenbergMultiple(self.road_length, self.lanes, 5, self.lag_parameter, self.entry_rate, self.multi_lane_rules)
        self.steps = 0
        self.data = np.empty((self.steps, self.lanes, self.road_length))

    def run(self, warmup_steps=0, steps=0, warmup_steps_mult=5, steps_mult=10):
        ws = warmup_steps_mult * self.road_length
        if warmup_steps:
            ws = warmup_steps

        s = ws * steps_mult
        if steps:
            s = steps
        self.steps = s

        start = time.time()
        mod = math.ceil(ws/100)
        for i in range(ws):
            if i % mod == 0:
                print(f"Warmup progress: {i / mod:3.0f}%",end="\r")
            self.model.step()
        end = time.time()
        print(f"Warmup progress: 100% - {end - start:.4f} seconds")

        start = time.time()
        mod = math.ceil(s/100)
        self.data = np.empty((self.steps, self.lanes, self.road_length))
        for i in range(s):
            if i % mod == 0:
                print(f"Simulation progress: {i / mod:3.0f}%",end="\r")
            self.data[i] = self.model.get_road()
            self.model.step()
        end = time.time()
        print(f"Simulation progress: 100% - {end - start:.4f} seconds")

    def reset(self):
        self.model = NagelSchreckenbergMultiple(self.road_length, self.lanes, 5, self.lag_parameter, self.entry_rate, self.multi_lane_rules)
        self.steps = 0
        self.data = np.empty((self.steps, self.lanes, self.road_length))

    def animate(self, interval=100, figsize=None, Multiple_color=False):
        if figsize is None:
            figsize = (10, self.lanes + 1)
        fig, ax = plt.subplots(figsize=figsize)
        
        color_dict = { 0: "#8B0000", 1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green" }
        if Multiple_color:
            color_dict = {}
        map_color = np.vectorize(lambda x: color_dict.get(x, "black"))

        self.plot_frame(ax, 0, map_color)

        def update(frame):
            self.plot_frame(ax, frame, map_color)
            return ax.patches

        ani = animation.FuncAnimation(fig, update, frames=self.steps, interval=interval)
        plt.show()

    def plot_frame(self, ax, frame, map_color):
        road_data = self.data[frame]
        relevant_road_data = np.where(road_data >= 0)
        x = relevant_road_data[1]
        y = relevant_road_data[0] + 1
        y = self.lanes - y # Inverse y, because array and barplot use inverse lane formats
        height = np.ones_like(y)

        ax.clear()
        ax.bar(x, height=height, bottom=y, width=1.0, color=map_color(road_data[road_data >= 0]), align="edge")

        ax.set_xlim(0, self.road_length)
        ax.set_ylim(0, self.lanes)
        ax.set_yticks([])
        ax.set_title("Nagel-Schreckenberg Traffic Flow (%s)" % frame)
        ax.set_xlabel("Road Position")

    def get_stats(self):
        time_stats = self.model.get_time_stats()

        if len(time_stats) == 0:
            return {
            "length": self.road_length * 7.5,
            "length_model": self.road_length,
            "time_min": 0,
            "time_max": 0,
            "time_avg": 0,
            "velocity_avg": 0
        }

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
    sim = SimulationMultiple(100, 4, 0.4, 1.8, True)
    sim.run()
    sim.print_stats()
    sim.animate(100)

    # stats = { "multi_lane_features": 0, "control": 0 }
    # sim = SimulationMultiple(200, 3, 0.3, 1.0, True)
    # for i in range(5):
    #     sim.run()
    #     stats["multi_lane_features"] += sim.get_stats()["time_avg"]
    #     sim.reset()
    # sim = SimulationMultiple(200, 3, 0.3, 1.0, False)
    # for i in range(5):
    #     sim.run()
    #     stats["control"] += sim.get_stats()["time_avg"]
    #     sim.reset()

    # print(f"Multi lane feature avg time: {stats['multi_lane_features']/5:.2f}")
    # print(f"Control avg time: {stats['control']/5:.2f}")
    