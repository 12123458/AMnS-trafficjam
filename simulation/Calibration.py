import numpy as np
import math
from concurrent.futures import ProcessPoolExecutor

from SimulationMultiple import SimulationMultiple

class Calibration:
    """This class serves to calibrate the parameters of a multi-lane Nagel-Schreckenberg model in order to achieve a given target travel time. 
    Due to the very skewed impact of the parameters, with entry_rate dominating heavily (as can be observed in a generated heatmap), only a small range of times is reliably able to be calibrated.
    Times that are outside the expected window (e.g. 20s-25s for 100 cells at 3 lanes) should be approached as scenarios instead.
    """
    def __init__(self, target_travel_time, convergence_range=1, road_length=100, lanes=3,
                 lag_parameter_min=0, lag_parameter_max=0.9, entry_rate_min=0.1, entry_rate_max=3, repetitions=10):
        """Initializes the calibration algorithm.
        
        Args:
            target_travel_time: Target average travel time to calibrate for.
            convergence_range: Allowed deviation from the target to accept as convergence. In time units.
            road_length: Number of cells.
            lanes: Number of lanes.
            lag_parameter_min: Lower bound of the calibration for the lag_parameter.
            lag_parameter_max: Upper bound of the calibration for the lag_parameter.
            entry_rate_min: Lower bound of the calibration for the entry_rate.
            entry_rate_max: Upper bound of the calibration for the entry_rate.
            repetitions: Number of repetitions to ensure consistent results and low variability.
        """
        self.target_travel_time = target_travel_time
        self.convergence_range = convergence_range
        self.road_length = road_length
        self.lanes = lanes
        self.repetitions = repetitions

        self.lag_parameter_min = lag_parameter_min
        self.lag_parameter_max = lag_parameter_max
        self.entry_rate_min = entry_rate_min
        self.entry_rate_max = entry_rate_max

    def calibrate(self, lag_parameter=None, entry_rate=None, lag_parameter_step=None, entry_rate_step=None, ratio=1):
        """Calibrates the parameters 'lag_parameter' and 'entry_rate' in order to achieve the target travel time.
        
        Args:
            lag_parameter: The initial lag_parameter to start calibration from. If not set, gets assigned 1/3 of the possible range.
            entry_rate: The initial entry_rate to start calibration from. If not set, gets assigned 1/2 of the possible range.
            lag_parameter_step: The initial lag_parameter_step to use for calibration. If not set, gets assigned 1/10 of the possible range.
            entry_rate_step: The initial entry_rate_step to use for calibration. If not set, gets assigned 1/10 of the possible range.
            ratio: The ratio of the parameter update steps relative to each other. Is only used if step sizes are not set manually. 1 means both are updated in the same relative steps, >1 means entry_rate is updated more, <1 means lag_rate is updated more.
        """
        # Set the initial parameters if not given
        if lag_parameter is None:
            lag_parameter = (self.lag_parameter_max - self.lag_parameter_min) / 3
        if entry_rate is None:
            entry_rate = (self.entry_rate_max - self.entry_rate_min) / 3
        
        # Define the update steps for the calibration
        if lag_parameter_step is None:
            lag_parameter_step = (self.lag_parameter_max - self.lag_parameter_min) / (10 * ratio)
        if entry_rate_step is None:
            entry_rate_step = (self.entry_rate_max - self.entry_rate_min) / (10 / ratio)
        
        dist = math.inf
        counter = 0 # Current iteration counter
        # Use a process pool to simultaneously simulate each repetition
        with ProcessPoolExecutor() as executor:
            while dist > self.convergence_range and counter < 100: # Hard break criterion at 100 iterations
                counter += 1

                sum_of_avg_travel_times = 0
                # Use the same parameters for each repetition
                indices = [(self.road_length, self.lanes, lag_parameter, entry_rate)] * self.repetitions
                # Calculate an average over the repetitions as a stable result
                for value in executor.map(Calibration.run_simulation, indices):
                    sum_of_avg_travel_times += value
                self.result_travel_time = sum_of_avg_travel_times / self.repetitions

                # Calculate distance and direction of the difference
                dist_old = dist
                dist = abs(self.result_travel_time - self.target_travel_time)
                is_larger = True if self.result_travel_time > self.target_travel_time else False

                # Save the current parameters as possible results; this assumes eventual convergence
                self.result_lag_parameter = lag_parameter
                self.result_entry_rate = entry_rate

                print(f"{counter}. Iteration with lag_parameter {lag_parameter:.3f} and entry_rate {entry_rate:.3f} results in {self.result_travel_time:.2f}s travel time ({dist:.2f}s too {"much" if is_larger else "little"})")

                sign_mult = -1 if is_larger else 1

                # Calculate the lag_parameter for the next iteration
                lag_parameter_old = lag_parameter
                lag_parameter += sign_mult * lag_parameter_step
                lag_parameter = max(min(lag_parameter, self.lag_parameter_max), self.lag_parameter_min)
                is_lag_parameter_edged = lag_parameter == lag_parameter_old

                # Calculate the entry_rate for the next iteration
                entry_rate_old = entry_rate
                entry_rate += sign_mult * entry_rate_step
                entry_rate = max(min(entry_rate, self.entry_rate_max), self.entry_rate_min)
                is_entry_rate_edged = entry_rate == entry_rate_old

                # With increasing iterations progressivly decrease the step sizes
                step_mult = (0.9 - 0.5*counter/100) if dist <= dist_old else (1.1 - 0.1*counter/100)
                lag_parameter_step *= step_mult
                lag_parameter_step = max(lag_parameter_step, (self.lag_parameter_max - self.lag_parameter_min) / 100) # Steps are atleast 1%
                entry_rate_step *= step_mult
                entry_rate_step = max(entry_rate_step, (self.entry_rate_max - self.entry_rate_min) / 100)

                # Check whether both parameters didn't change
                # This can only happen if we are in a corner and want to move even further into the corner
                # Because both parameters are strictly monotonic, no danger of local optima
                if is_lag_parameter_edged and is_entry_rate_edged:
                    print("Reached both edges -> Can't improve result")
                    break

    @staticmethod
    def run_simulation(args):
        """Run a simulation. Because no instance variables are needed @staticmethod can be used instead of using tricks with __call__."""
        road_length, lanes, lag_parameter, entry_rate = args
        sim = SimulationMultiple(road_length, lanes, lag_parameter, entry_rate)
        sim.run(verbose=False)
        return sim.get_stats()["time_avg"]
    
    def print_results(self):
        """Print the final result."""
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
    cal.calibrate(lag_parameter_step=0.05, entry_rate_step=0.01)
    cal.print_results()