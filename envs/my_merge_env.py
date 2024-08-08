from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.envs.highway_env import HighwayEnv, Vehicle
from highway_env.envs.common.action import Action

class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 1440,
            "screen_height": 240,
            "centering_position": [0.3, 0.5],
            "scaling": 5,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 1,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 10,  # default=200
            "HIGH_SPEED_REWARD": 0.1,  # default=0.5
            "HEADWAY_COST": 1,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 0.1,  # default=4
            "traffic_density": 1,  # easy or hard modes
            "vehicles_count": 3,
            "vehicles_density": 1.2,
            "normalize_reward": False,
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        ends = [150, 80, 80, 150]
        # the optimal reward is 0
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(ends[:3])) ** 2 / (
                    10 * ends[2]))
        else:
            Merging_lane_cost = 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        # Headway_cost = np.log(
        #     headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 # + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [-self.config["COLLISION_REWARD"] - self.config["HEADWAY_COST"] - self.config["MERGING_LANE_COST"],
                                 self.config["HIGH_SPEED_REWARD"] ],
                                [0, 1])
        return reward

    def _regional_reward(self):
        ends = [150, 80, 80, 150]
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def _cost(self, action):
        return sum(self._agent_cost(action, vehicle) for vehicle in self.controlled_vehicles) \
            / len(self.controlled_vehicles)

    def _agent_cost(self, action, vehicle):
        Headway_cost = 0
        # compute headway cost
        # if action != 0 or action != 2:
        headway_distance = self._compute_headway_distance(vehicle)
        end_distance = 380 - vehicle.position[0]
        distance = min(headway_distance, end_distance)
        if distance <= 80:
            Headway_cost1 = -np.log(distance / (1.2 * vehicle.speed)) if vehicle.speed > 0 else 0

        # else:
        hd_f = 80
        hd_r = 80
        for v in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 1) and v.lane_index[2] == 0:
                if v.position[0] > vehicle.position[0]:
                    hd_f = min(hd_f, v.position[0] - vehicle.position[0])
                else:
                    hd_r = min(hd_r,vehicle.position[0] - v.position[0])

        distance = min(80, (hd_f + hd_r))
        if distance <= 80:
            Headway_cost2 = -np.log(distance / (1.2 * vehicle.speed)) if vehicle.speed > 0 else 0

        Headway_cost = Headway_cost1 + Headway_cost2
        Headway_cost = utils.lmap(Headway_cost, [-2, 5], [0, 1])

        return Headway_cost



    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.time += 1
        self._clear_vehicles()
        self._spawn_vehicle()

        # info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        # for v in self.controlled_vehicles:
        #     agent_info.append([v.position[0], v.position[1], v.speed])
        # info["agents_info"] = agent_info

        # for vehicle in self.controlled_vehicles:
        #     vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        # info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        # self._regional_reward()
        # info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        # obs = np.asarray(obs).reshape((len(obs), -1))
        # obs = obs.reshape(-1)
        return obs, reward, terminated, truncated, info

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 350) #or self.time > 100

    def _is_truncated(self, action) -> list:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        cost = []
        # cost.append(self._cost(action))
        if bool(self.vehicle.position[0] > 350 and not self.vehicle.crashed):
            cost.append(1)
        else:
            cost.append(0)
        return cost

    def _info(self) -> np.ndarray:

        return self.observation_type.observe_info()

    def _reset(self) -> None:
        self.time = 0
        self._make_road()
        self._make_vehicles()
        self.action_is_safe = True
        # self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [sum(ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c",StraightLane([sum(ends[:2]), 0], [sum(ends[:3]), 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([sum(ends[:3]), 0], [sum(ends), 0], line_types=[c, c]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(50 + np.random.normal() * 10, 0),
                                                     speed=15 + np.random.normal() * 2)
        road.vehicles.append(ego_vehicle)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        space = 0
        for _ in range(self.config["vehicles_count"]):
            # lane_id = np.random.randint(2)
            lane_id = 0
            lane = road.network.get_lane(("a", "b", lane_id))
            # vehicle = other_vehicles_type.create_random(road,
            #                                             lane_from="a",
            #                                             lane_to="b",
            #                                             lane_id=lane_id,
            #                                             speed=15 + self.np_random.normal() * 1,#lane.speed_limit,
            #                                             spacing=1 / self.config["vehicles_density"],
            #                                             ).plan_route_to("d")

            # vehicle.randomize_behavior()
            # road.vehicles.append(vehicle)
            ab = np.random.uniform(0, 1)
            if ab > 0.4: # default 0.2
                road.vehicles.append(
                        other_vehicles_type(road, road.network.get_lane(("a", "b", lane_id)).position(
                            np.random.randint(0, 40) + space, 0), speed=15 + np.random.normal() * 1.5))
            space += 45

        bc = np.random.uniform(0, 1)
        if bc > 0.6:
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("b", "c", 0)).position(
                    np.random.randint(20, 80), 0), speed=15 + np.random.normal() * 1.5))
        cd = np.random.uniform(0, 1)
        if cd > 0.2:
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("c", "d", 0)).position(
                    np.random.randint(20, 150), 0), speed=15 + np.random.normal() * 1.5))
            # vehicle.enable_lane_change = False


        # ramping obstacle car
        if np.random.uniform(0, 1) > 0.8:
            merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                np.random.randint(110, 150), 0), speed=15 + np.random.normal() * 1.5)
            # merging_v.target_speed = 30
            road.vehicles.append(merging_v)

        self.vehicle = ego_vehicle

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.4, # default 0.6
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        # route = self.np_random.choice(range(4), size=2, replace=False)  #随机生成目的地车道

        # route[1] = (route[0] + 2) % 4 if go_straight else route[1]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        # lane_id = np.random.randint(2)
        lane_id = 0
        vehicle = vehicle_type.make_on_lane(self.road, ("a" , "b", lane_id),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=15 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 25:
                return
        vehicle.plan_route_to("d")
        vehicle.enable_lane_change = False
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or vehicle.position[0] < 380 - 4 * vehicle.LENGTH]

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

class MergeEnvContinuous(AbstractEnv):
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False},
            "controlled_vehicles": 1,
            "screen_width": 1200,
            "screen_height": 240,
            "centering_position": [0.3, 0.5],
            "scaling": 5,
            "simulation_frequency": 50,  # [Hz]
            "duration": 100,  # time step
            "policy_frequency": 50,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 0.5,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_COST": 4,  # default=4
            "traffic_density": 0.6,  # easy or hard modes
            "vehicles_count": 5,
            "vehicles_density": 1.0,
            "normalize_reward": False,
        })
        return config

    def _reward(self, action) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        return self._reward1(action)
        # return self._reward2(action)
        r_ef = 0
        # if self.vehicle.speed < 10:
        #     r_ef = self.vehicle.speed - 10
        # elif self.vehicle.speed > 20:
        #     r_ef = - self.vehicle.speed + 10




        # if headway_distance > 1.2 * self.vehicle.speed + 2 * self.vehicle.LENGTH:
        #     r_safty = (1.2 * self.vehicle.speed + 2 * self.vehicle.LENGTH - headway_distance) / 10
        # r_safty = np.log(
        #     headway_distance / (self.config["HEADWAY_TIME"] * (self.vehicle.speed)))
        r_safty = -max(self._compute_headway_risk(headway_distance), self._compute_merging_risk())
        # elif headway_distance < 1.2 * self.vehicle.speed:
        #     r_safty = 200 * (headway_distance - 1.2 * self.vehicle.speed)

        # r_gap = 0
        # if self.vehicle.position[0] <= 198:
        #     merging_time_gap = self._compute_merging_time()
        #     # print("merging_time_gap:", merging_time_gap)
        #     if merging_time_gap < 48 / self.vehicle.speed:
        #         r_gap = merging_time_gap - (48 / self.vehicle.speed)
            # print("r_gap:", r_gap)

        # if r_safty > 0:
        #     r_safty = 0
        # if headway_distance != 80 and self.vehicle.speed > 0:
        #     Headway_reward = np.log(headway_distance / (1.2 * self.vehicle.speed))

        # r_goal = 0
        # if self.vehicle.position[0] >= 300 and not self.vehicle.crashed :
        #     r_goal = 20

        # r_ramp = 0
        # if self.vehicle.position[0] > 230:
            # for v in self.road.vehicles:
            #     if v.lane_index[2] == 0 and isinstance(v, ControlledVehicle):
            #         r_ramp += (v.target_speed - v.speed) / v.target_speed
            # r_ramp = -abs(15 - self.vehicle.speed) / 15
        # r_ramp *= -0.5


        # print("collision:", r_collision)
        # print("high_speed:", self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1))
        # print("merging:", r_ramp)
        # print("safty:", r_safty)


        # print("total reward:", reward)
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [-self.config["COLLISION_REWARD"] - self.config["HEADWAY_COST"] - self.config["MERGING_LANE_COST"],
        #                          self.config["HIGH_SPEED_REWARD"] ],
        #                         [0, 1])
        # reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        # return utils.lmap(reward,
        #                   [self.config["collision_reward"] + self.config["safty_distance_reward"],
        #                    self.config["high_speed_reward"]  + self.config["goal_reward"]],
        #                   [0, 1])
        # reward = utils.lmap(reward, [-200, 200], [0, 1])
        return reward

    def _reward1(self, action):
        # efficiency
        r_ef = self.vehicle.speed / 50
        # r_ef = self.vehicle.speed / 10
        # comfortable
        r_a = 0
        acceleration = float(action)
        if acceleration > 1.47 or acceleration < -2:
            r_a = -0.01

        # collision
        r_collision = 0
        for v in self.road.vehicles:
            if v.crashed:
                r_collision = -20

        # target lane
        r_right_lane = 0
        # if self.vehicle.lane_index == ("c", "d", 0):
        #     r_right_lane = 0.1

        return r_ef + r_a + r_collision + r_right_lane

    def _reward2(self, action):
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        r_collision = 0
        for v in self.road.vehicles:
            if v.crashed:
                r_collision = -1

        r_safty = 0
        headway_distance = self._compute_headway_distance(self.vehicle)
        r_safty = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * (self.vehicle.speed)))

        r_merge = 0
        if self.vehicle.position[0] <= 198:
            r_ramp = -np.exp(-(self.vehicle.position[0] - 198) ** 2 / (10 * 148))

        return self.config["COLLISION_REWARD"] * r_collision \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_COST"] * r_merge \
                 + self.config["HEADWAY_COST"] * r_safty

    def _reward3(self, action):
        r_ef = self.vehicle.speed / 50

        r_collision = 0
        for v in self.road.vehicles:
            if v.crashed:
                r_collision = -1

        r_safty = 0
        headway_distance = self._compute_headway_distance(self.vehicle)
        r_safty = -max(self._compute_headway_risk(headway_distance), self._compute_merging_risk())

        r_merge = 0
        if self.vehicle.position[0] <= 198:
            r_ramp = -np.exp(-(self.vehicle.position[0] - 198) ** 2 / (10 * 148))
    def _compute_merging_risk(self):
        dist = 100
        relative_speed = 0
        for v in self.road.vehicles:
            if v.position[0] < self.vehicle.position[0]:
                if self.vehicle.position[0] - v.position[0] < dist:
                    dist = self.vehicle.position[0] - v.position[0]
                    relative_speed = abs(self.vehicle.speed - v.speed)
        risk = 0
        if dist < 5:
            risk = 1.0  # 碰撞风险最大
        else:
            risk = np.exp(-dist / (2 * relative_speed + 0.001))
        return risk
    def _compute_headway_risk(self, hd):
        risk = 0
        if hd < 5:
            risk = 1.0
        else:
            risk = np.exp(-hd / (1.2 * self.vehicle.speed))
        return risk

    def _cost(self, action):
        c_collision = 0
        if self.vehicle.crashed:
            c_collision = -10

        c_safty = 0
        headway_distance = self._compute_headway_distance(self.vehicle)
        if headway_distance != 80 and self.vehicle.speed > 0:
            c_safty = np.log(headway_distance / (1.2 * self.vehicle.speed))

        # c_efficiency = self.vehicle.speed / 500

        c_efficienct = 0
        if self.vehicle.speed > 10 and self.vehicle.speed < 20:
            c_efficienct = self.vehicle.speed / 50

        c_comfortable = 0
        acceleration = float(action)
        if acceleration > 1.47 or acceleration < -4:
            c_comfortable = -0.1

        cost = c_collision + c_comfortable + c_safty
        return cost
    # def _rewards(self, action: str) -> Dict[Text, float]:
    #     # ttc = self.observation_type.ttc  # 前车车距
    #     # r_s = 0
    #     # t = abs(self.observation_type.ttc_to_end_h - self.observation_type.ttc_to_end_l)
    #     # if ttc <= 1.2:
    #     #     r_s += np.exp(-ttc / 10)
    #     # if t <= 1:
    #     #     r_s += np.exp(-t / 10)
    #
    #     # if self.observation_type.v <= 30 and self.observation_type.v >= 5:
    #     #     r_ef = np.clip((self.observation_type.v - 10) / 50, 0, 1)
    #     r_ef = self.observation_type.v / 50
    #
    #     r_a = 0
    #     acceleration = float(action)
    #     if acceleration > 1.47 or acceleration < -2:
    #         r_a = -0.1
    #
    #     return {
    #         "collision_reward": 0.5 * (self.vehicle.crashed or self.time > 100),
    #         "high_speed_reward": r_ef,
    #         # "safty_distance_reward": r_s,
    #         "drastic acceleration penalty": r_a,
    #         # "merging_speed_reward": sum(  # Altruistic penalty
    #         #     (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
    #         #     for vehicle in self.road.vehicles
    #         #     if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)),
    #         "goal_reward": 0.2 * bool(self.vehicle.position[0] >= 300 and not self.vehicle.crashed)
    #     }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        crashed = False
        for v in self.road.vehicles:
            if v.crashed:
                crashed = True
        return crashed or bool(self.vehicle.position[0] > 220) or self.time > 2000

    def _is_truncated(self) -> bool:
        if bool(self.vehicle.position[0] > 220 and not self.vehicle.crashed):
            return True
        return False

    def _reset(self) -> None:
        self.time = 0
        self._make_road()
        self._make_vehicles()
        # self.last_state = self.reset()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool]:
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self.change_obs_format_9()
        # obs = self.change_obs_format_15()
        reward = self._reward(action)
        terminated = self._is_terminated()
        success = self._is_truncated()
        # cost = self._cost(action)
        self.time += 1
        self._clear_vehicles()
        self._spawn_vehicle()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, success

    def change_obs_format_9(self):
        v0 = self.vehicle.speed
        # a0 = (v0 - np.linalg.norm(self.last_state[3], self.last_state[4])) / 0.02
        a0 = self.vehicle.action['acceleration']
        x0 = self.vehicle.position[0]
        y0 = self.vehicle.position[1]
        df = 100
        v1 = 0
        d1 = 0
        a1 = 0
        for v in self.road.vehicles:
            if v.position[0] > x0:
                if v.position[0] - x0 < df:
                    df = v.position[0] - x0
                    d1 = np.linalg.norm(np.array(v.position) - np.array(self.vehicle.position))
                    v1 = v.speed
                    a1 = v.action['acceleration']
        a2 = 0
        v2 = 0
        d2 = 0
        dr = 100
        for v in self.road.vehicles:
            if v.position[0] < x0:
                if x0 - v.position[0] < dr:
                    dr = x0 - v.position[0]
                    d2 = np.linalg.norm(np.array(v.position) - np.array(self.vehicle.position))
                    v2 = v.speed
                    a2 = v.action['acceleration']
        return [a0, v0, y0, a1, v1, d1, a2, v2, d2]

    def change_obs_format_15(self):
        v0 = self.vehicle.speed
        a0 = self.vehicle.action['acceleration']
        x0 = self.vehicle.position[0]
        y0 = self.vehicle.position[1]

        vf = None
        df = 100
        for v in self.road.vehicles:
            if v.position[0] > x0:
                if v.position[0] - x0 < df:
                    df = v.position[0] - x0
                    vf = v
        if vf:
            a1 = vf.action['acceleration']
            v1 = vf.speed
            x1 = vf.position[0]
            y1 = vf.position[1]

            d1 = np.linalg.norm(np.array(vf.position) - np.array(self.vehicle.position))
        else:
            a1 = 0
            v1 = 0
            x1 = 0
            y1 = 0
            d1 = 0

        vff = None
        dff = 100
        for v in self.road.vehicles:
            if v.position[0] > x1:
                if v.position[0] - x1 < dff:
                    dff = v.position[0] - x1
                    vff = v
        if vff:
            a2 = vff.action['acceleration']
            v2 = vff.speed
            x2 = vff.position[0]
            y2 = vff.position[1]

            d2 = np.linalg.norm(np.array(vff.position) - np.array(self.vehicle.position))
        else:
            a2 = 0
            v2 = 0
            x2 = 0
            y2 = 0
            d2 = 0

        vr = None
        dr = 100
        for v in self.road.vehicles:
            if v.position[0] < x0:
                if x0 - v.position[0] < dr:
                    dr = x0 - v.position[0]
                    vr = v

        if vr:
            a3 = vr.action['acceleration']
            v3 = vr.speed
            x3 = vr.position[0]
            y3 = vr.position[1]

            d3 = np.linalg.norm(np.array(vr.position) - np.array(self.vehicle.position))
        else:
            a3 = 0
            v3 = 0
            x3 = 0
            y3 = 0
            d3 = 0

        vrr = None
        drr = 100
        for v in self.road.vehicles:
            if v.position[0] < x3:
                if x3 - v.position[0] < drr:
                    dr = x3 - v.position[0]
                    vrr = v
        if vrr:
            a4 = vrr.action['acceleration']
            v4 = vrr.speed
            x4 = vrr.position[0]
            y4 = vrr.position[1]

            d4 = np.linalg.norm(np.array(vrr.position) - np.array(self.vehicle.position))
        else:
            a4 = 0
            v4 = 0
            x4 = 0
            y4 = 0
            d4 = 0

        return [a0, v0, y0, a1, v1, d1, a2, v2, d2, a3, v3, d3, a4, v4, d4]

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        # y = [0, 5]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        # net = RoadNetwork.straight_road_network(2, start=0,
        #                                         length=1000, nodes_str=("a", "b"))
        # net = RoadNetwork.straight_road_network(3 + 1, start=500,
        #                                         length=100, nodes_str=("b", "c"), net=net)
        # net = RoadNetwork.straight_road_network(3, start=1000,
        #                                         length=200,
        #                                         nodes_str=("c", "d"), net=net)
        # for i in range(2):
        #     net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
        #     net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
        #     net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))
        # 汇入前的3车道线
        # net.add_lane("a", "b", StraightLane([0, -4], [150 + 80, -4], line_types=[c, s]))
        net.add_lane("a", "b", StraightLane([0, 0], [150 , 0], line_types=[c, c]))
        # net.add_lane("a", "b", StraightLane([0, 4], [150 + 80, 4], line_types=[c, c]))

        # 与汇入车道结合处
        # net.add_lane("b", "c", StraightLane([150 + 80, -4], [150 + 80 + 80, -4], line_types=[c, c], forbidden=True))
        net.add_lane("b", "c", StraightLane([150, 0], [150 + 48, 0], line_types=[c, s], forbidden=True))
        # net.add_lane("b", "c", StraightLane([150 + 80, 4], [150 + 80 + 48, 4], line_types=[c, s], forbidden=False))
        # 汇入车道后车道
        # net.add_lane("c", "d", StraightLane([150 + 80 + 80, -4], [150 + 80 + 80 + 150, -4], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([150 + 48, 0], [150 + 100, 0], line_types=[c, c], forbidden=True))
        # net.add_lane("c", "d", StraightLane([150 + 80 + 48, 4], [150 + 80 + 150, 4], line_types=[c, c]))

        # Merging lane
        amplitude = 3.25
        # ljk = StraightLane([0, 6.5 + 4 + 4], [150, 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        ljk = StraightLane([50, 15], [150, 5], line_types=[c, c], forbidden=True)
        # lkb = StraightLane([0, 6 + 4 + 4], [150 + 80, 8], line_types=[c, c], forbidden=True)
        lkc = StraightLane([150, 5], [150 + 48, 0], line_types=[n, c], forbidden=True)

        # lkb = SineLane(ljk.position(150, -amplitude), ljk.position(230, -amplitude),
        #                amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        # lbc = SineLane(ljk.position(sum(ends[:2]), -amplitude), ljk.position(sum(ends[:2]), -amplitude - 4),
        #                amplitude, -np.pi / 4, -np.pi / 4, line_types=[c, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        # net.add_lane("k", "l", lkb)
        net.add_lane("k", "c", lkc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(0, 0),
                                                     speed=15)

        road.vehicles.append(ego_vehicle)
        # ego_vehicle.target_speed = 30

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(5, 0), speed=31.5))

        for _ in range(self.config["vehicles_count"]):
            lane_id = 0
            lane = road.network.get_lane(("a", "b", lane_id))
            vehicle = other_vehicles_type.create_random(road,
                                                        lane_from="a",
                                                        lane_to="b",
                                                        lane_id=lane_id,
                                                        speed=15 + self.np_random.normal() * 1,#lane.speed_limit,
                                                        spacing=1 / self.config["vehicles_density"],
                                                        ).plan_route_to("d")
        # for _ in range(2):
        #     lane_id = np.random.randint(2)
        #     vehicle = other_vehicles_type.create_random(road,
        #                                                 lane_from="b",
        #                                                 lane_to="c",
        #                                                 lane_id=lane_id,
        #                                                 speed=15 + self.np_random.normal() * 1,  # lane.speed_limit,
        #                                                 spacing=1 / self.config["vehicles_density"],
        #                                                 ).plan_route_to("d")
        #     vehicle = other_vehicles_type.create_random(road,
        #                                                 lane_from="c",
        #                                                 lane_to="d",
        #                                                 lane_id=lane_id,
        #                                                 speed=15 + self.np_random.normal() * 1,  # lane.speed_limit,
        #                                                 spacing=1 / self.config["vehicles_density"],
        #                                                 ).plan_route_to("d")

            vehicle.enable_lane_change = False
            # vehicle.randomize_behavior()
            road.vehicles.append(vehicle)


        if np.random.uniform(0, 1) > 1:
            merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(150, 0), speed=20)
            # merging_v.target_speed = 30
            road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

    # def _make_vehicles(self, n_vehicles: int = 10) -> None:
    #     """
    #     Populate a road with several vehicles on the highway and on the merging lane
    #
    #     :return: the ego-vehicle
    #     """
    #     # Configure vehicles
    #     vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
    #     vehicle_type.COMFORT_ACC_MAX = 6
    #     vehicle_type.COMFORT_ACC_MIN = -3
    #
    #     # Random vehicles
    #     simulation_steps = 3
    #     for t in range(n_vehicles - 1):
    #         self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
    #     for _ in range(simulation_steps):
    #         [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]
    #
    #     # Challenger vehicle
    #     self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)
    #
    #     # Controlled vehicles
    #     self.controlled_vehicles = []
    #     for ego_id in range(0, 1):
    #         ego_lane = self.road.network.get_lane(("j", "k", 0))
    #         ego_vehicle = self.action_type.vehicle_class(
    #                          self.road,
    #                          ego_lane.position(60 + 5*self.np_random.normal(1), 0),
    #                          speed=ego_lane.speed_limit,
    #                          heading=ego_lane.heading_at(60))
    #         try:
    #             ego_vehicle.plan_route_to("d")
    #             ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
    #             ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
    #         except AttributeError:
    #             pass
    #
    #         self.road.vehicles.append(ego_vehicle)
    #         self.controlled_vehicles.append(ego_vehicle)
    #         for v in self.road.vehicles:  # Prevent early collisions
    #             if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
    #                 self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        # route = self.np_random.choice(range(4), size=2, replace=False)  #随机生成目的地车道

        # route[1] = (route[0] + 2) % 4 if go_straight else route[1]

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        lane_id = 0
        vehicle = vehicle_type.make_on_lane(self.road, ("a" , "b", lane_id),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=15 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 30:
                return
        vehicle.plan_route_to("d")
        vehicle.enable_lane_change = False
        # vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        # is_leaving = lambda vehicle: "c" in vehicle.lane_index[0] and "c" in vehicle.lane_index[1] \
        #                              >= vehicle.lane.length - 4 * vehicle.LENGTH
        # self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
        #                       vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]
        # is_leaving = lambda vehicle: self.road.vehicles.position[0] < 380
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or vehicle.position[0] < 250 - 4 * vehicle.LENGTH]
    # def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
    #     return "d" in vehicle.lane_index[0] \
    #            and "d" in vehicle.lane_index[1] \
    #            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance