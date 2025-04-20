import numpy as np
import matplotlib.pyplot as plt
import time
import math

from NagelSchreckenbergMultiple import NagelSchreckenbergMultiple

class ScenarioRushour:
    """This class functions similar to SimulationMultiple, but is specifically tailored to model a rushhour scenario. 
    More generally, this class allows for modeling dynamically changing lag_parameter and entry_rate to simulate changing traffic conditions.
    """
    def __init__(self, road_length=100, lanes=3, lag_parameter=0.3, entry_rate=0.5):
        self.road_length = road_length
        self.lanes = lanes
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate

        self.model = NagelSchreckenbergMultiple(self.road_length, self.lanes, 5, self.lag_parameter, self.entry_rate)
        self.data = []
        self.avg_travel_times = []

    def _warmup(self, warmup_steps=0, warmup_steps_mult=5, verbose=True):
        ws = warmup_steps_mult * self.road_length
        if warmup_steps:
            ws = warmup_steps

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
        self.model.reset_queue()
        
    def _run(self, steps, verbose=True):
        start = time.time()
        mod = math.ceil(steps/100)
        for i in range(steps):
            if verbose and i % mod == 0:
                print(f"Simulation progress: {i / mod:3.0f}%",end="\r")
            self.data.append(self.model.get_road())
            self.model.step()
        end = time.time()
        if verbose:
            print(f"Simulation progress: 100% - {end - start:.4f} seconds")

    def _calculate_parameters(self, duration, parameters):
        """This helper method supplements a list of changed traffic parameter tuples by adding additional tuples for normal conditions.
        
        Args:
            duration: The total duration of the scenario
            parameters: List of tuples in the form (start_time, end_time, lag_parameter, entry_rate), which characterize traffic conditions for a given interval.
        """
        parameters = sorted(parameters, key=lambda x: x[0])

        if parameters:
            # Add default-value tuples in-between
            for i in range(1, len(parameters)):
                first_entry_start = parameters[0][0]
                last_entry_end = parameters[-1][1]

                start = parameters[i-1][1]
                end = parameters[i][0]
                if end - start > 0:
                    parameters.append((start, end, self.lag_parameter, self.entry_rate))
            
            # Add default-value tuples at the start and end, if necessary
            if first_entry_start > 0:
                parameters.append((0, first_entry_start, self.lag_parameter, self.entry_rate))
            if last_entry_end < duration:
                parameters.append((last_entry_end, duration, self.lag_parameter, self.entry_rate))
        else:
            # In case of no parameters, simply add a single default entry
            parameters.append((0, duration, self.lag_parameter, self.entry_rate))

        # Sort the parameters by start time
        return sorted(parameters, key=lambda x: x[0])

    def run(self, duration=43200, parameters=[]):
        """Run the scenario for the given duration and parameters.

        Args:
            duration: The total duration of the scenario
            parameters: List of tuples in the form (start_time, end_time, lag_parameter, entry_rate), which characterize traffic conditions for a given interval.
        """
        parameters = self._calculate_parameters(duration, parameters)

        self._warmup()

        duration_in_minutes = 43200 / 60
        # Run the simulation with the chosen parameters at any given time
        for start, end, lag_parameter, entry_rate in parameters:
            self.model.set_parameters(lag_parameter, entry_rate)
            for i in range(start, end):
                # Track stats each minute
                if i % 60 == 0:
                    if self.model.get_time_stats().any():
                        self.avg_travel_times.append(np.average(self.model.get_time_stats()))
                    self.model.reset_stats()
                    print(f"Simulating minute: {(i / 60) + 1:.0f} of {duration_in_minutes:.0f}",end="\r")
                self.data.append(self.model.get_road())
                self.model.step()
        print("\nFinished simulation")


if __name__ == "__main__":
    scen = ScenarioRushour(lag_parameter=0.2, entry_rate=0.2)

    rush_hour_schedule = [
        (9, 10, 0.2, 0.4),     # Early rise
        (10, 10.5, 0.3, 0.9),  # Picking up
        (10.5, 11, 0.4, 1.5),  # Getting busy
        (11, 11.5, 0.5, 2.0),  # Near peak
        (11.5, 12, 0.6, 2.4),  # Peak rush hour
        (12, 12.5, 0.6, 2.1),  # Holding high
        (12.5, 13, 0.5, 1.7),  # Decline begins
        (13, 13.5, 0.4, 1.2),  # Easing off
        (13.5, 14, 0.3, 0.6),  # Low intensity
    ]

    scen.run(parameters=[(int((start-8)*3600), int((end-8)*3600), lag, entry) for start, end, lag, entry in rush_hour_schedule])

    x_positions = list(range(0, 721, 60))
    x_labels = ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

    plt.plot(scen.avg_travel_times)
    plt.xticks(ticks=x_positions, labels=x_labels)
    plt.xlabel("Hour")
    plt.ylabel("Travel time in seconds")
    plt.title("Average travel time over 12 hours for 750m road")
    plt.grid(True)
    plt.show()