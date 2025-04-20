import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import time
import math

from NagelSchreckenbergMultiple import NagelSchreckenbergMultiple

# Create legend elements as a constant
colors = ["#8B0000", "red", "orange", "yellow", "lightgreen", "green"]
labels = ["0 speed", "1 speed", "2 speed", "3 speed", "4 speed", "5 speed"]
legend_elements = [Line2D([0], [0], color=color, lw=4, label=label) for color, label in zip(colors, labels)]

class SimulationMultiple:
    """This class uses a Nagel-Schreckenberg model for multiple lanes in order to simulate traffic, display some statistics and show an representative animation"""
    def __init__(self, road_length=100, lanes=2, lag_parameter=0.3, entry_rate=0.5, multi_lane_rules=True):
        """Initialize the simulation.
        
        Args:
            road_length: The amount of cells representing the road. Each cell corresponds to 7.5m.
            lanes: The number of lanes.
            lag_parameter: The dawdling factor used in rule 3.
            entry_rate: The number of cars entering (or at least, trying to) the road each second. As, at most, one car can enter on each lane each second, the maximum value is equal to the number of lanes.
            multi_lane_rules: A controlling parameter to compare results with. Disabling it results in essentially multiple single lane models next to each other.
        """
        self.road_length = road_length
        self.lanes = lanes
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate
        self.multi_lane_rules = multi_lane_rules

        # For the simulation we already know that we only work with a maximum speed of 5, as is common for Nagel-Schreckenberg models
        self.model = NagelSchreckenbergMultiple(self.road_length, self.lanes, 5, self.lag_parameter, self.entry_rate, self.multi_lane_rules)
        self.steps = 0                                                      # The number of steps used in the simulation; primarily needed for the length of the animation
        self.data = np.empty((self.steps, self.lanes, self.road_length))    # The state of the road over all time steps

    def run(self, warmup_steps=0, steps=0, warmup_steps_mult=5, steps_mult=10, verbose=True):
        """Runs the simulation.
        
        Args:
            warmup_steps: The number of steps to use for initially running the model so that it reaches a natural initial state instead of using an arbitrary initial car density.
            steps: The number of steps to actually run the simulation and capture data for.
            warmup_steps_mult: If warmup_steps is not set, this calculates the number of warmup_steps based on the length of the road.
            steps_mult: If steps is not set, this calculates the number of steps based on the number of warmup_steps.
            verbose: Whether to display the progress of the simulation.
        """
        # Calculate the number of warmup steps based on road length...
        ws = warmup_steps_mult * self.road_length
        # ...if it is not set manually
        if warmup_steps:
            ws = warmup_steps

        # Calculate the number of steps based on the number of warmup steps...
        s = ws * steps_mult
        # ...if it is not set manually
        if steps:
            s = steps
        # Remember the number of steps for later, e.g. the animation
        self.steps = s

        # Run the model through the warmup steps and optionally display the current progress
        start = time.time()
        mod = math.ceil(ws/100)
        for i in range(ws):
            if verbose and i % mod == 0:
                print(f"Warmup progress: {i / mod:3.0f}%",end="\r")
            self.model.step()
        end = time.time()
        if verbose:
            print(f"Warmup progress: 100% - {end - start:.4f} seconds")
        
        # Reset runtime state of the model after warmup
        self.model.reset_stats()
        self.model.reset_queue()

        # run the model through the steps and optionally display the current progress
        start = time.time()
        mod = math.ceil(s/100)
        self.data = np.empty((self.steps, self.lanes, self.road_length))
        for i in range(s):
            if verbose and i % mod == 0:
                print(f"Simulation progress: {i / mod:3.0f}%",end="\r")
            # Also save the current road state at each step
            self.data[i] = self.model.get_road()
            self.model.step()
        end = time.time()
        if verbose:
            print(f"Simulation progress: 100% - {end - start:.4f} seconds")

    def reset(self):
        """Reset the simulation in case it should be run multiple times with different run configurations"""
        self.model = NagelSchreckenbergMultiple(self.road_length, self.lanes, 5, self.lag_parameter, self.entry_rate, self.multi_lane_rules)
        self.steps = 0
        self.data = np.empty((self.steps, self.lanes, self.road_length))

    def animate(self, interval=100, figsize=None, multiple_color=True):
        """Animates each step of the completed simulation in the given interval.
        
        Args:
            interval: Time between each state. Given in ms.
            figsize: Size of the plot.
            multiple_color: Whether to use a different color for each velocity or simply black and white.
        """
        if figsize is None:
            figsize = (10, self.lanes + 1)
        fig, ax = plt.subplots(figsize=figsize)
        
        color_dict = { 0: "#8B0000", 1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green" }
        if not multiple_color:
            color_dict = {}
        map_color = np.vectorize(lambda x: color_dict.get(x, "black"))

        self._plot_frame(ax, 0, map_color)

        def update(frame):
            self._plot_frame(ax, frame, map_color)
            return ax.patches

        ani = animation.FuncAnimation(fig, update, frames=self.steps, interval=interval)
        plt.show()

    def _plot_frame(self, ax, frame, map_color):
        """Helper method to generate the representation of the current road state using a bar diagram"""
        road_data = self.data[frame]
        relevant_road_data = np.where(road_data >= 0)
        x = relevant_road_data[1]
        y = relevant_road_data[0] + 1
        y = self.lanes - y # Inverse y, because array and barplot use inverse lane formats; so row 0 in the data matrix corresponds to the first/left most lane
        height = np.ones_like(y)

        ax.clear()
        ax.bar(x, height=height, bottom=y, width=1.0, color=map_color(road_data[road_data >= 0]), align="edge")

        ax.set_xlim(0, self.road_length)
        ax.set_ylim(0, self.lanes)
        ax.set_yticks([])
        ax.set_title("Nagel-Schreckenberg Traffic Flow (%s)" % frame)
        ax.set_xlabel("Road Position")
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0, 1), ncol=1, frameon=False)

    def get_stats(self):
        """Calculate some stats based on the captured time data or from model parameters"""
        time_stats = self.model.get_time_stats()

        return {
            "length": self.road_length * 7.5,
            "length_model": self.road_length,
            "time_min": np.min(time_stats),
            "time_max": np.max(time_stats),
            "time_avg": np.average(time_stats),
            "velocity_avg": self.road_length * 7.5 / np.average(time_stats),
            "cars": len(time_stats),
        }
    
    def print_stats(self):
        """Print the stats for simple feedback"""
        stats = self.get_stats()
        print(f"Total road segment length: {stats["length"]:.0f}m")
        print(f"Average completion time: {stats["time_avg"]:.2f}s")
        print(f"Average speed: {stats["velocity_avg"]:.2f}m/s or {stats["velocity_avg"]*3.6:.0f}km/h")

if __name__ == "__main__":
    sim = SimulationMultiple(100, 3, 0.479, 1.045, True)
    sim.run()
    sim.print_stats()
    sim.animate(100)
    