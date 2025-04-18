import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor

from SimulationMultiple import SimulationMultiple

class Calibration:
    def __init__(self, target_travel_time, convergence_range=1, road_length=100, lanes=3,
                 lag_parameter_min=0, lag_parameter_max=0.9, entry_rate_min=0.1, entry_rate_max=3, repetitions=10):
        self.target_travel_time = target_travel_time
        self.convergence_range = convergence_range
        self.road_length = road_length
        self.lanes = lanes
        self.repetitions = repetitions

        self.lag_parameter_min = lag_parameter_min
        self.lag_parameter_max = lag_parameter_max
        self.entry_rate_min = entry_rate_min
        self.entry_rate_max = entry_rate_max

    def calibrate(self, lag_parameter=None, entry_rate=None):
        # Set the initial parameters if not given
        if lag_parameter is None:
            # Initialize the lag_parameter smaller, because it makes sense in the modeled system to not dawdle too much
            lag_parameter = (self.lag_parameter_max - self.lag_parameter_min) / 3
        if entry_rate is None:
            entry_rate = (self.entry_rate_max - self.entry_rate_min) / 2
        
        # Define the update steps for the calibration
        lag_parameter_step = (self.lag_parameter_max - self.lag_parameter_min) / 10
        entry_rate_step = (self.entry_rate_max - self.entry_rate_min) / 10
        
        dist = math.inf
        counter = 0
        with ProcessPoolExecutor() as executor:
            while dist > self.convergence_range and counter < 100: # Hard break criterion
                counter += 1

                sum_of_avg_travel_times = 0
                indices = [(self.road_length, self.lanes, lag_parameter, entry_rate)] * self.repetitions
                for value in executor.map(Calibration.run_simulation, indices):
                    sum_of_avg_travel_times += value
                self.result_travel_time = sum_of_avg_travel_times / self.repetitions

                dist_old = dist
                dist = abs(self.result_travel_time - self.target_travel_time)
                is_larger = True if self.result_travel_time > self.target_travel_time else False

                self.result_lag_parameter = lag_parameter
                self.result_entry_rate = entry_rate

                print(f"{counter}. Iteration with lag_parameter {lag_parameter:.3f} and entry_rate {entry_rate:.3f} results in {self.result_travel_time:.2f}s travel time ({dist:.2f}s too {"much" if is_larger else "little"})")

                sign_mult = -1 if is_larger else 1

                lag_parameter_old = lag_parameter
                lag_parameter += sign_mult * lag_parameter_step
                lag_parameter = max(min(lag_parameter, self.lag_parameter_max), self.lag_parameter_min)
                is_lag_parameter_edged = lag_parameter == lag_parameter_old

                entry_rate_old = entry_rate
                entry_rate += sign_mult * entry_rate_step
                entry_rate = max(min(entry_rate, self.entry_rate_max), self.entry_rate_min)
                is_entry_rate_edged = entry_rate == entry_rate_old

                # With increasing iterations increasingly decrease the step sizes
                step_mult = (0.9 - 0.5*counter/100) if dist <= dist_old else (1.1 - 0.1*counter/100)
                lag_parameter_step *= step_mult
                lag_parameter_step = max(lag_parameter_step, (self.lag_parameter_max - self.lag_parameter_min) / 100) # Steps are atleast 1%
                entry_rate_step *= step_mult
                entry_rate_step = max(entry_rate_step, (self.entry_rate_max - self.entry_rate_min) / 100)

                # This simple edge check is allowed, because both parameters are strictly monotonic, so no danger of local optima
                if is_lag_parameter_edged and is_entry_rate_edged:
                    print("Reached both edges -> Can't improve result")
                    break

    @staticmethod
    def run_simulation(args):
        road_length, lanes, lag_parameter, entry_rate = args
        sim = SimulationMultiple(road_length, lanes, lag_parameter, entry_rate)
        sim.run(verbose=False)
        return sim.get_stats()["time_adj"]
    
    def print_results(self):
        print("\nFinal Parameters:")
        print(f"    lag_parameter: {self.result_lag_parameter:.3f}")
        print(f"    entry_rate: {self.result_entry_rate:.3f}")
        print("Results:")
        road_length = self.road_length * 7.5
        print(f"    Road length: {road_length}m")
        print(f"    Target travel time: {self.target_travel_time:.2f}s")
        target_speed = road_length / self.target_travel_time
        print(f"    Target speed: {target_speed:.2f}m/s | {target_speed*3.6:.2f}km/h")
        print(f"    Result travel time: {self.result_travel_time:.2f}s")
        result_speed = road_length / self.result_travel_time
        print(f"    Result speed: {result_speed:.2f}m/s | {result_speed*3.6:.2f}km/h")
    
if __name__ == "__main__":
    cal = Calibration(25, 1)
    cal.calibrate()
    cal.print_results()