import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor

from SimulationMultiple import SimulationMultiple

class ParameterHeatmap:
    def __init__(self, road_length=100, lanes=3, 
                 lag_parameter_min=0, lag_parameter_max=0.9, lag_parameter_steps=10, 
                 entry_rate_min=0.1, entry_rate_max=3, entry_rate_steps=10,
                 multi_lane_rules=True, repetitions=2):
        """Initializes the traffic model heatmap generation with parameter ranges and settings.

        Args:
            road_length (int, optional): Length of the simulated road. Defaults to 100.
            lanes (int, optional): Number of lanes in the road. Defaults to 3.
            lag_parameter_min (float, optional): Minimum lag parameter. Defaults to 0.
            lag_parameter_max (float, optional): Maximum lag parameter. Defaults to 0.9.
            lag_parameter_steps (int, optional): Number of steps between min and max lag. Defaults to 10.
            entry_rate_min (float, optional): Minimum entry rate of vehicles. Defaults to 0.1.
            entry_rate_max (float, optional): Maximum entry rate of vehicles. Defaults to 3.
            entry_rate_steps (int, optional): Number of steps between min and max entry rate. Defaults to 10.
            multi_lane_rules (bool, optional): Whether the additional multi-lane rules are enabled. Defaults to True. For debugging/comparison purposes
            repetitions (int, optional): Number of times to repeat each simulation run. Defaults to 2.
        """
        self.road_length = road_length
        self.lanes = lanes
        self.lag_parameter_min = lag_parameter_min
        self.lag_parameter_max = lag_parameter_max
        self.lag_parameter_steps = lag_parameter_steps
        self.entry_rate_min = entry_rate_min
        self.entry_rate_max = entry_rate_max
        self.entry_rate_steps = entry_rate_steps
        self.multi_lane_rules = multi_lane_rules
        self.repetitions = repetitions

        self.lag_parameters = np.linspace(lag_parameter_min, lag_parameter_max, lag_parameter_steps)
        self.entry_rates = np.linspace(entry_rate_min, entry_rate_max, entry_rate_steps)
        self.max_simulations = lag_parameter_steps * entry_rate_steps * repetitions

        self.avg_travel_times = np.zeros((lag_parameter_steps, entry_rate_steps))

    def run_simulations(self):
        print(f"Generation progress: 0 / {self.max_simulations}", end="\r")

        indices = [(i, lag, j, entry) for (i, lag), (j, entry) in product(enumerate(self.lag_parameters), enumerate(self.entry_rates))]
        with ProcessPoolExecutor() as executor:
            for i, j, value in executor.map(self, indices):
                self.avg_travel_times[i, j] = value

        print(f"Generation progress: {self.max_simulations} / {self.max_simulations}")

    # Defined as __call__ so the ProcessPoolExecutor works
    def __call__(self, args):
        i, lag_parameter, j, entry_rate = args
        sum_of_avg_travel_time = 0
        for r in range(self.repetitions):
            sim = SimulationMultiple(road_length=self.road_length, lanes=self.lanes, lag_parameter=lag_parameter,
                                    entry_rate=entry_rate, multi_lane_rules=self.multi_lane_rules)
            sim.run(verbose=False)
            current_run = i*self.entry_rate_steps*self.repetitions + j*self.repetitions + (r+1)
            print(f"Generation progress: {current_run} / {self.max_simulations}", end="\r")
            sum_of_avg_travel_time += sim.get_stats()["time_adj"]
        return i, j, sum_of_avg_travel_time / self.repetitions

    def plot(self, figsize=(10,8), save_path=None):
        plt.figure(figsize=figsize)
        sns.heatmap(self.avg_travel_times, annot=True, fmt=".2f", cmap="viridis_r",
                    xticklabels=np.round(self.entry_rates, 2),
                    yticklabels=np.round(self.lag_parameters, 2))

        plt.xlabel("Entry Rate")
        plt.ylabel("Lag Parameter")
        plt.title("Simulation Metric Heatmap")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

if __name__ == "__main__":
    ph = ParameterHeatmap(road_length=100, lanes=3, repetitions=3, multi_lane_rules=True)
    ph.run_simulations()
    ph.plot(save_path="heatmap_road100_lanes3.png")