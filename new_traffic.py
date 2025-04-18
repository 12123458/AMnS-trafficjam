import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
import random
import pandas as pd
from io import StringIO

class TrafficModel:
    def __init__(self, road_length, max_speed, lag_param, entry_rate, vehicle_density, lanes=2):
        self.road_length = road_length
        self.max_speed = max_speed
        self.lag_param = lag_param
        self.entry_rate = entry_rate
        self.vehicle_density = vehicle_density
        self.lanes = lanes
        self.vehicle_id_counter = 1
        self.vehicle_entry_times = {}
        self.vehicle_exit_times = []
        self.reset()

    def reset(self):
        total_cells = self.road_length * self.lanes
        num_vehicles = int(self.vehicle_density * total_cells)
        self.road = np.zeros((self.lanes, self.road_length), dtype=int)
        self.speeds = np.zeros((self.lanes, self.road_length), dtype=int)
        self.ids = np.zeros((self.lanes, self.road_length), dtype=int)
        self.vehicle_id_counter = 1
        self.vehicle_entry_times = {}
        self.vehicle_exit_times = []

        positions = np.random.choice(total_cells, num_vehicles, replace=False)
        for pos in positions:
            lane = pos // self.road_length
            idx = pos % self.road_length
            self.road[lane, idx] = 1
            self.speeds[lane, idx] = np.random.randint(1, self.max_speed + 1)
            self.ids[lane, idx] = self.vehicle_id_counter
            self.vehicle_entry_times[self.vehicle_id_counter] = 0
            self.vehicle_id_counter += 1

    def step(self, current_step):
        new_road = np.zeros_like(self.road)
        new_speeds = np.zeros_like(self.speeds)
        new_ids = np.zeros_like(self.ids)

        for lane in range(self.lanes):
            for i in range(self.road_length):
                if self.road[lane, i] == 1:
                    speed = self.speeds[lane, i]
                    vid = self.ids[lane, i]

                    if random.random() < 0.05:
                        speed = max(speed - 1, 0)

                    distance = 0
                    for d in range(1, speed + 1):
                        if self.road[lane, (i + d) % self.road_length] == 0:
                            distance += 1
                        else:
                            break

                    if distance < speed:
                        speed = distance
                    else:
                        speed = min(speed + 1, self.max_speed)

                    new_pos = (i + speed)
                    if new_pos >= self.road_length:
                        self.vehicle_exit_times.append(current_step - self.vehicle_entry_times.get(vid, 0))
                    else:
                        new_road[lane, new_pos] = 1
                        new_speeds[lane, new_pos] = speed
                        new_ids[lane, new_pos] = vid

        for lane in range(self.lanes):
            if random.random() < self.entry_rate and new_road[lane, 0] == 0:
                new_road[lane, 0] = 1
                new_speeds[lane, 0] = random.randint(1, self.max_speed)
                new_ids[lane, 0] = self.vehicle_id_counter
                self.vehicle_entry_times[self.vehicle_id_counter] = current_step
                self.vehicle_id_counter += 1

        self.road = new_road
        self.speeds = new_speeds
        self.ids = new_ids

    def simulate(self, steps):
        self.vehicle_exit_times = []
        self.vehicle_entry_times = {}
        for step in range(steps):
            self.step(step)
        return np.mean(self.vehicle_exit_times) if self.vehicle_exit_times else 0

    def simulate_multiple_runs(self, steps, runs=10):
        times = []
        for _ in range(runs):
            self.reset()
            times.append(self.simulate(steps))
        return np.mean(times)

    def calibrate(self, target_travel_time, steps=100):
        def objective(x):
            self.max_speed = int(x[0])
            self.lag_param = int(x[1])
            self.reset()
            avg_time = self.simulate_multiple_runs(steps, runs=3)
            return abs(avg_time - target_travel_time)

        result = minimize(objective,
                          [self.max_speed, self.lag_param],
                          bounds=[(1, 10), (0, 5)],
                          method='L-BFGS-B')
        self.max_speed, self.lag_param = int(result.x[0]), int(result.x[1])
        return self.max_speed, self.lag_param

    def dynamic_simulation(self, density_list, real_times, steps=100):
        for density, target_time in zip(density_list, real_times):
            print(f"\n📊 Calibrating for Vehicle Density: {density:.2f}")
            self.vehicle_density = density
            calibrated_speed, calibrated_lag = self.calibrate(target_time, steps)
            print(f"Calibrated Max Speed: {calibrated_speed}")
            print(f"Calibrated Lag Param: {calibrated_lag}")
            print("Validating...")
            sim_time = self.simulate_multiple_runs(steps, runs=5)
            print(f"Simulated Avg Travel Time: {sim_time:.2f} | Target: {target_time:.2f} | Difference: {abs(sim_time - target_time):.2f}")

    def face_validation(self, real_data, simulation_steps=100):
        print("\n🔍 Running Face Validation...")
        sim_time = self.simulate_multiple_runs(simulation_steps, runs=5)
        real_avg_time = np.mean(real_data['travel_time'])

        print(f"Simulated Avg Travel Time: {sim_time:.2f}")
        print(f"Real Data Avg Travel Time: {real_avg_time:.2f}")
        print(f"Difference: {abs(sim_time - real_avg_time):.2f}")

    def visualize(self, steps=100):
        self.reset()
        fig, ax = plt.subplots()
        scat = ax.scatter([], [], c='black', s=50)
        ax.set_xlim(0, self.road_length)
        ax.set_ylim(-1, self.lanes)
        ax.set_title("Traffic Simulation (2 Lanes)")

        def update(frame):
            self.step(frame)
            xs, ys = [], []
            for lane in range(self.lanes):
                for i in range(self.road_length):
                    if self.road[lane, i] == 1:
                        xs.append(i)
                        ys.append(lane + 0.5)
            scat.set_offsets(np.c_[xs, ys])
            return scat,

        ani = FuncAnimation(fig, update, frames=steps, blit=True, interval=10)
        plt.show()

# === Sample Real-World Data Inline ===
real_data_csv = """
travel_time
19.8
29.5
30.0
23.4
31.2
30.7
29.9
"""
real_data = pd.read_csv(StringIO(real_data_csv))

# === Parameters and Initialization ===
road_length = 100
initial_max_speed = 10
initial_lag_param = 2
entry_rate = 0.2
vehicle_density = 0.3
target_travel_time = 20

model = TrafficModel(road_length,
                     initial_max_speed,
                     initial_lag_param,
                     entry_rate,
                     vehicle_density,
                     lanes=2)

# === Calibration ===
print("\n🔧 Starting Calibration...")
calibrated_speed, calibrated_lag = model.calibrate(target_travel_time)
print("✅ Calibration Completed!")
print(f"Calibrated Max Speed: {calibrated_speed}")
print(f"Calibrated Lag Parameter: {calibrated_lag}")

# === Face Validation ===
model.face_validation(real_data, simulation_steps=100)

# === Dynamic Simulation with Real Data ===
# Example density and real average times from dataset
densities = [0.4, 0.6, 0.8]
real_times = [22.0, 25.0, 28.0]
model.dynamic_simulation(densities, real_times, steps=100)

# === Visualization ===
model.visualize(steps=100)
