import numpy as np
import random
import math

class NagelSchreckenbergMultiple:
    """This class represents a traffic model with multiple lanes using an extended Nagel-Schreckenberg model"""
    def __init__(self, road_length=100, lanes=2, v_max=5, lag_parameter=0.3, entry_rate=0.5, multi_lane_rules=True):
        """Initialize the model.
        
        Args:
            road_length: The amount of cells representing the road. Each cell corresponds to 7.5m.
            lanes: The number of lanes.
            v_max: The maximum velocity. This implicitly sets how fast one unit of velocity is in the real world. v_max=5 means 5 cells per second, so 37.5m/s.
            lag_parameter: The dawdling factor used in rule 3.
            entry_rate: The number of cars entering (or at least, trying to) the road each second. As, at most, one car can enter on each lane each second, the maximum value is equal to the number of lanes.
            multi_lane_rules: A controlling parameter to compare results with. Disabling it results in essentially multiple single lane models next to each other.
        """
        self.road_length = road_length
        self.lanes = lanes
        self.v_max = v_max
        self.p = lag_parameter
        self.entry_rate = entry_rate
        self.multi_lane_rules = multi_lane_rules

        self.road = np.full((self.lanes, self.road_length), -1)     # Represents the current state of the road; -1 = empty, -2 = closed, otherwise = car ID
        self.velocities = {}                                        # The speed of the current cars
        
        self.next_car_id = 0                                        # Next id for car creation
        self.entry_queue = []                                       # Cars waiting to enter the road due to congestion at the start

        self.current_step = 0                                       # Current time
        self.start_times = {}                                       # Captures the time that cars entered the model at
        self.times_taken = {}                                       # Records the total time a given car needed to leave the model

        self.closing_end_times = {}                                 # Controls when closed lanes are opened again

    def step(self):
        """This method calculates the changes occuring during a single second passing"""
        new_road = np.full((self.lanes, self.road_length), -1)      # Changes to the car states are calculated based on the current road and applied to a new road
        new_velocities = {}                                         # Only capture the velocities of currently existing cars

        # If a lane is still closed, apply the closure
        for lane, end_time in self.closing_end_times.items():
            if self.current_step < end_time:
                new_road[lane, :] = -2

        # Get all cars from the road; the indices are automatically ordered by lane and then position
        car_indices = np.argwhere(self.road >= 0)
        for lane, pos in car_indices:
            car_id = self.road[lane, pos]
            v = self.velocities[car_id]
            new_lane = lane                                         # Due to multi-lane behaviors cars may switch the lane

            # 1. Acceleration up to maximum speed
            v = min(v + 1, self.v_max)
            v_target = v                                            # For multi-lane checks, remember what speed a car would like to reach

            # 2. Slowing down due to gap
            # Slicing the relevant road segment out of the matrix leads to performance gains
            left_lane = max(lane-1, 0)
            right_lane = min(lane+1, self.lanes-1)
            relevant_lane = lane - left_lane
            pos_back = max(0, pos - self.v_max)
            pos_diff = pos - pos_back
            relevant_road_slice = self.road[left_lane:right_lane+1, pos_back:(pos+v+1)]

            # The gap is only relevant if it slows the car down, and the car is not already able to leave the model
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
                # The min is a safeguard for the edge of the model
                for i in range(min(pos_diff + v + 1, relevant_road_slice.shape[1])):
                    if relevant_road_slice[relevant_right_lane, i] != -1:
                        lane_free = False
                        break
                # Only switch the lane if it is possible
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
            else: # If the car leaves the model, record the time
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
                # Popping the first/oldest car -> FIFO queue
                car_id = self.entry_queue.pop(0)
                new_road[lane, 0] = car_id
                # New cars enter with max speed as they are either free to speed up anyway or would slow down anyway
                new_velocities[car_id] = self.v_max

        self.road = new_road
        self.velocities = new_velocities
        self.current_step += 1

    def get_road(self):
        """Get the current state of the road, but replace the car-ids with their velocity"""
        return np.vectorize(lambda x: self.velocities.get(x, -1) if x >= 0 else x)(self.road)
    
    def get_time_stats(self):
        """Get an array of the time taken data, ignoring the car-id"""
        return np.array(list(self.times_taken.values()))
    
    def reset_stats(self):
        """Resets the time taken data to get fresh readings"""
        self.times_taken = {}

    def reset_queue(self):
        """Reset the queue, which is useful after a warmup period to ensure a clean model state"""
        self.entry_queue = []

    def set_parameters(self, lag_parameter, entry_rate):
        """Set the parameters even after initialization to facilitate scenario tests"""
        self.lag_parameter = lag_parameter
        self.entry_rate = entry_rate
    
    def close_lane(self, lane, duration):
        """Close a lane for a given duration to facilitate scenario tests"""
        self.closing_end_times[lane] = self.current_step + duration
        # Just close a lane and effectively delete all the cars on it
        # In the grand scheme this shouldn't impact the results much
        self.road[lane, :] = -2 