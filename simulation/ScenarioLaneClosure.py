import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math

from NagelSchreckenbergMultiple import NagelSchreckenbergMultiple

class ScenarioLaneClosure:
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
        parameters = sorted(parameters, key=lambda x: x[0])

        if parameters:
            for i in range(1, len(parameters)):
                first_entry_start = parameters[0][0]
                last_entry_end = parameters[-1][1]

                start = parameters[i-1][1]
                end = parameters[i][0]
                if end - start > 0:
                    parameters.append((start, end, self.lag_parameter, self.entry_rate))
            
            
            
            if first_entry_start > 0:
                parameters.append((0, first_entry_start, self.lag_parameter, self.entry_rate))
            if last_entry_end < duration:
                parameters.append((last_entry_end, duration, self.lag_parameter, self.entry_rate))
        else:
            parameters.append((0, duration, self.lag_parameter, self.entry_rate))

        return sorted(parameters, key=lambda x: x[0])

    def run(self, duration=43200, parameters=[]):
        parameters = sorted(parameters, key=lambda x: x[0])

        self._warmup()

        duration_in_minutes = 43200 / 60
        cur_parameter = parameters.pop(0) if parameters else (-1,-1,-1)
        for step in range(duration):
            if step == cur_parameter[0]:
                self.model.close_lane(cur_parameter[2], cur_parameter[1])
                if parameters:
                    cur_parameter = parameters.pop(0)
            if step % 60 == 0:
                if self.model.get_time_stats().any():
                    self.avg_travel_times.append(np.average(self.model.get_time_stats()))
                self.model.reset_stats()
                print(f"Simulating minute: {(step / 60) + 1:.0f} of {duration_in_minutes:.0f}",end="\r")
            self.data.append(self.model.get_road())
            self.model.step()
        print("\nFinished simulation")


if __name__ == "__main__":
    scen = ScenarioLaneClosure(lag_parameter=0.3, entry_rate=1.0)

    scen.run(parameters=[(3*3600, 3*3600, 2)])

    x_positions = list(range(0, 721, 60))
    x_labels = ["8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]

    plt.plot(scen.avg_travel_times)
    plt.xticks(ticks=x_positions, labels=x_labels)
    plt.xlabel("Hour")
    plt.ylabel("Travel time in seconds")
    plt.title("Average travel time over 12 hours for 750m road")
    plt.grid(True)
    plt.show()