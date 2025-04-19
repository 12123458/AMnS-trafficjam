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

    def run(self, warmup_steps=0, steps=0, warmup_steps_mult=5, steps_mult=10, verbose=True):
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
            if verbose and i % mod == 0:
                print(f"Warmup progress: {i / mod:3.0f}%",end="\r")
            self.model.step()
        end = time.time()
        if verbose:
            print(f"Warmup progress: 100% - {end - start:.4f} seconds")
        
        self.model.reset_stats()

        start = time.time()
        mod = math.ceil(s/100)
        self.data = np.empty((self.steps, self.lanes, self.road_length))
        for i in range(s):
            if verbose and i % mod == 0:
                print(f"Simulation progress: {i / mod:3.0f}%",end="\r")
            self.data[i] = self.model.get_road()
            self.model.step()
        end = time.time()
        if verbose:
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
            "velocity_avg": 0,
            "cars": 0,
            "time_avg_by_cars": 0,
            "velocity_avg_by_cars": 0,
            "time_adj": 0,
            "velocity_avg_adj": 0
        }

        # Our method of measuring time has the crucial weakness that it only records cars that complete the track
        # However, with high lag_parameters cars tend to block the entrance of the road and therefore no more travel, which distorts the data
        reference_speed = 4 + (1-self.lag_parameter)
        reference_cars = self.steps * self.entry_rate
        cars = len(time_stats)
        time_estimated = self.road_length / ((cars / reference_cars) * reference_speed)

        time_avg = np.average(time_stats)

        # Interpolate a time based on the recorded and estimated times
        k = time_avg * 0.5
        t = abs(time_estimated - time_avg) / (abs(time_estimated - time_avg) + k)
        time_adj = time_avg * (1-t) + time_estimated * t
        # The estimation is only necessary if the recorded time is unrealistically low
        # If the estimation is lower than the recording, it suggests that the recording is fine as it is
        if time_estimated < time_avg:
            time_adj = time_avg

        return {
            "length": self.road_length * 7.5,
            "length_model": self.road_length,
            "time_min": np.min(time_stats),
            "time_max": np.max(time_stats),
            "time_avg": time_avg,
            "velocity_avg": self.road_length * 7.5 / time_avg,
            "cars": cars,
            "time_avg_by_cars": time_estimated,
            "velocity_avg_by_cars": self.road_length * 7.5 / time_estimated,
            "time_adj": time_adj,
            "velocity_avg_adj": self.road_length * 7.5 / time_adj,
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"Total road segment length: {stats["length"]:.0f}m")
        print(f"Average completion time (recoded): {stats["time_avg"]:.2f}s")
        print(f"Average completion time (estimated): {stats["time_avg_by_cars"]:.2f}s")
        print(f"Average completion time (interpolated): {stats["time_adj"]:.2f}s")
        print(f"Average speed (recoded): {stats["velocity_avg"]:.2f}m/s or {stats["velocity_avg"]*3.6:.0f}km/h")
        print(f"Average speed (estimated): {stats["velocity_avg_by_cars"]:.2f}m/s or {stats["velocity_avg_by_cars"]*3.6:.0f}km/h")
        print(f"Average speed (interpolated): {stats["velocity_avg_adj"]:.2f}m/s or {stats["velocity_avg_adj"]*3.6:.0f}km/h")

if __name__ == "__main__":
    sim = SimulationMultiple(100, 3, 0.33, 1.548, True)
    sim.run()
    sim.print_stats()
    sim.animate(100)
    