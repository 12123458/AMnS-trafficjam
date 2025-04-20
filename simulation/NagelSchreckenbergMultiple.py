import numpy as np
import random
import math

class NagelSchreckenbergMultiple:
    # entry_rate makes only sense in interval [0; lanes]
    def __init__(self, road_length=100, lanes=2, v_max=5, lag_parameter=0.3, entry_rate=0.5, multi_lane_rules=True):
        self.road_length = road_length
        self.lanes = lanes
        self.v_max = v_max
        self.p = lag_parameter
        self.entry_rate = entry_rate
        self.multi_lane_rules = multi_lane_rules

        # -1 = empty, otherwise = car ID
        self.road = np.full((self.lanes, self.road_length), -1)
        self.velocities = {}
        
        self.next_car_id = 0

        self.current_step = 0
        self.start_times = {}
        self.times_taken = {}

        self.closing_end_times = {}

        self.entry_queue = []

    def step(self):
        new_road = np.full((self.lanes, self.road_length), -1)
        new_velocities = {}

        for lane, end_time in self.closing_end_times.items():
            if self.current_step < end_time:
                new_road[lane, :] = -2

        car_indices = np.argwhere(self.road >= 0)
        for lane, pos in car_indices:
            car_id = self.road[lane, pos]
            v = self.velocities[car_id]
            new_lane = lane

            # 1. Acceleration
            v = min(v + 1, self.v_max)
            v_target = v

            # 2. Slowing down due to gap
            left_lane = max(lane-1, 0)
            right_lane = min(lane+1, self.lanes-1)
            relevant_lane = lane - left_lane
            pos_back = max(0, pos - self.v_max)
            pos_diff = pos - pos_back
            relevant_road_slice = self.road[left_lane:right_lane+1, pos_back:(pos+v+1)]

            gap = 1
            while gap <= v and ((pos + gap) >= self.road_length or relevant_road_slice[relevant_lane, pos_diff+gap] == -1):
                gap += 1
            v = min(v, gap - 1)

            # Overtaking
            # Check first whether we need to break and if there is a left lane 
            if self.multi_lane_rules and v < v_target and lane != left_lane:
                # Drivers already on the lane have priority so we need sufficient free space
                relevant_left_lane = relevant_lane - 1
                lane_free = True
                for i in range(pos_diff+1):
                    # Don't overtake if left traffic is close
                    # Could possibly be further refined by checking the left car's speed
                    if relevant_road_slice[relevant_left_lane, i] != -1:
                        lane_free = False
                        break
                # If the lane is free, check whether we gain anything by switching
                if lane_free:
                    gap = 1
                    while gap <= v_target and ((pos + gap) >= self.road_length or relevant_road_slice[relevant_left_lane, pos_diff+gap] == -1):
                        gap += 1
                    v_overtake = gap - 1
                    # Only switch if the speed on the new lane is higher
                    if v_overtake > v:
                        v = v_overtake
                        new_lane = left_lane

            # Merging
            # Check whether we already are on the rightmost lane and whether we already switched
            if self.multi_lane_rules and lane != right_lane and new_lane == lane:
                # Drivers already on the lane have priority so we need sufficient free space
                lane_free = True
                relevant_right_lane = relevant_lane + 1
                for i in range(min(pos_diff + v + 1, relevant_road_slice.shape[1])):
                    if relevant_road_slice[relevant_right_lane, i] != -1:
                        lane_free = False
                        break
                if lane_free:
                    new_lane = right_lane

            # 3. Slowing down due to chance
            if v > 0 and random.random() < self.p:
                v -= 1

            # 4. Move
            new_pos = (pos + v)
            if new_pos < self.road_length:
                new_road[new_lane, new_pos] = car_id
                new_velocities[car_id] = v
            else:
                start_time = self.start_times.get(car_id)
                if start_time is not None:
                    self.times_taken[car_id] = self.current_step - start_time

        # New cars entering
        # Fill new cars initially into a queue to keep track of how long they have to wait outside of our road due to a congestion at the start
        # This ensures that the entry_rate parameter never becomes useless and distorts values
        remaining_entry_rate = self.entry_rate
        while remaining_entry_rate > 0:
            if random.random() < remaining_entry_rate:
                self.entry_queue.append(self.next_car_id)
                self.start_times[self.next_car_id] = self.current_step
                self.next_car_id += 1
            remaining_entry_rate -= 1

        # Randomly place the new cars in the lanes if there is space
        for lane in np.random.permutation(self.lanes):
            if self.entry_queue and new_road[lane, 0] == -1:
                car_id = self.entry_queue.pop(0)
                new_road[lane, 0] = car_id
                # New cars don't start standing, but a certain speed
                new_velocities[car_id] = random.randint(self.v_max, self.v_max)

        self.road = new_road
        self.velocities = new_velocities
        self.current_step += 1

    def get_road(self):
        return np.vectorize(lambda x: self.velocities.get(x, -1) if x >= 0 else x)(self.road)
    
    def get_time_stats(self):
        return np.array(list(self.times_taken.values()))
    
    def reset_stats(self):
        self.times_taken = {}

    def reset_queue(self):
        self.entry_queue = []

    def set_parameters(self, lag_parameter, entry_rate):
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate
    
    def close_lane(self, lane, duration):
        self.closing_end_times[lane] = self.current_step + duration
        # Just close a lane and effectively delete all the cars on it
        # In the grand scheme this shouldn't impact the results much
        self.road[lane, :] = -2 