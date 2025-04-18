import numpy as np
import random
import math

class NagelSchreckenbergSingle:
    def __init__(self, road_length=100, car_density=0.2, v_max=5, lag_parameter=0.3, entry_rate=0.5):
        self.road_length = road_length
        self.v_max = v_max
        self.p = lag_parameter
        self.entry_rate = entry_rate

        self.num_cars = int(road_length * car_density)
        self.road = [-1] * road_length  # -1 = empty, otherwise = car ID
        self.velocities = {}

        positions = random.sample(range(road_length), self.num_cars)
        for i, pos in enumerate(positions):
            self.road[pos] = i 
            self.velocities[i] = random.randint(math.ceil(v_max/2.0), v_max)
        
        self.next_car_id = self.num_cars

        self.current_step = 0
        self.time_taken = {}

    def step(self):
        new_road = [-1] * self.road_length
        new_velocities = {}

        car_positions = [(i, self.road[i]) for i in range(self.road_length) if self.road[i] != -1]

        for i, car_id in car_positions:
            v = self.velocities[car_id]

            # 1. Acceleration
            v = min(v + 1, self.v_max)

            # 2. Slowing down due to gap
            d = 1
            while d <= v and ((i + d) >= self.road_length or self.road[(i + d)] == -1):
                d += 1
            v = min(v, d - 1)

            # 3. Slowing down due to chance
            if v > 0 and random.random() < self.p:
                v -= 1

            # 4. Move
            new_pos = (i + v)
            if new_pos < self.road_length:
                new_road[new_pos] = car_id
                new_velocities[car_id] = v
            else:
                start_time = self.time_taken.get(car_id)
                if start_time is not None:
                    self.time_taken[car_id] = self.current_step - start_time

        # New cars entering
        if new_road[0] == -1 and random.random() < self.entry_rate:
            new_road[0] = self.next_car_id
            # New cars don't start standing, but a certain speed
            new_velocities[self.next_car_id] = random.randint(math.ceil(self.v_max/2.0), self.v_max)
            self.time_taken[self.next_car_id] = self.current_step # Record start time
            self.next_car_id += 1

        self.road = new_road
        self.velocities = new_velocities

        self.current_step += 1

    def get_road(self):
        return np.array([self.velocities[x] if x != -1 else -1 for x in self.road])
    
    def get_time_stats(self):
        ids_to_remove = {i for i in self.road if i != -1}
        return np.array([v for k, v in self.time_taken.items() if k not in ids_to_remove])

