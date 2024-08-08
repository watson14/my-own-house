from typing import Dict, Text, Tuple

import numpy as np
import os

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class lhpHighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics1"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            # "collision_reward": -1,    # The reward received when colliding with a vehicle.
            # "right_lane_reward": 0.,  # The reward received when driving on the right-most lanes, linearly mapped to
            #                            # zero for other lanes.
            # "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            #                            # lower speeds according to config["reward_speed_range"].
            # "lane_change_reward": 0,   # The reward received at each lane change action.
            "safety_reward": None,
            "efficiency_reward": None,
            "comfortable_reward": None,
            "reward_speed_range": [20, 30],
            "normalize_reward": False,       # 正则化

            "offroad_terminal": False,
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._step = 0
        self.vx_old = 25
        self.vy_old = 0

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        self._step += 1
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)

        return obs, reward, terminated, truncated, info

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # rewards = self._rewards(action)
        # reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        # if self.config["normalize_reward"]:
        #     reward = utils.lmap(reward,
        #                         [self.config["collision_reward"],
        #                          self.config["high_speed_reward"] + self.config["right_lane_reward"]],
        #                         [0, 1])
        # reward *= rewards['on_road_reward']
        
        alpha = 1
        beta = 0.4
        gamma = 0.1
        r_cl, r_ef, r_cf = self._rewards(action)
        # print(r_cl, "\n", r_ef, "\n", r_cf, "\n")
        reward = alpha * r_cl + beta * r_ef + gamma * r_cf
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [-120,
                                 10],
                                [0, 1])
        # print(reward,"\n")
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        # 用于计算高速奖励
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        # 安全性
        r_cl = 0
        d1 = self.observation_type.f0_dis  # 前车车距
        if action == 0:
            d2 = self.observation_type.l0_dis  # 目标车道前车距离
            d3 = self.observation_type.l1_dis  # 目标车道后车距离
        elif action == 2:
            d2 = self.observation_type.r0_dis  # 目标车道前车距离
            d3 = self.observation_type.r1_dis  # 目标车道后车距离
        else:
            d2 = d1
            d3 = d1

        D_min = min(d1, d2, d3)
        threshold = 20
        if self.vehicle.crashed:
            r_cl = -100
        elif D_min <= threshold:
            r_cl = -1 / D_min

        # 效率
        r_ef = 0
        v_obs = min(self.observation_type.f0_speed,
                 self.observation_type.l0_speed,
                 self.observation_type.l1_speed,
                 self.observation_type.r0_speed,
                 self.observation_type.r1_speed)
        v_maxL = 30
        v_minL = 20
        w_s = 0.5
        w_d = 1
        step = self._step
        dura = 50
        if forward_speed < v_obs:
            r_ef = - ((v_obs - forward_speed) / (v_maxL - v_minL)) ** 2
        else:
            r_ef = w_s * ((forward_speed - v_minL) / (v_maxL - v_minL)) ** 2 + w_d * (step / dura) ** 2

        # 舒适度
        a_x = self.config['simulation_frequency'] * (forward_speed - self.vx_old)
        a_y = self.config['simulation_frequency'] * (self.observation_type.vy - self.vy_old)
        r_cf = - np.exp((forward_speed - v_minL) / (v_maxL - v_minL)) * abs(a_y) - abs(a_x)

        self.vx_old = forward_speed
        self.vy_old = self.observation_type.vy

        return r_cl, r_ef, r_cf

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


# class HighwayEnvFast(HighwayEnv):
#     """
#     A variant of highway-v0 with faster execution:
#         - lower simulation frequency
#         - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
#         - only check collision of controlled vehicles with others
#     """
#     @classmethod
#     def default_config(cls) -> dict:
#         cfg = super().default_config()
#         cfg.update({
#             "simulation_frequency": 5,
#             "lanes_count": 3,
#             "vehicles_count": 20,
#             "duration": 30,  # [s]
#             "ego_spacing": 1.5,
#         })
#         return cfg
#
#     def _create_vehicles(self) -> None:
#         super()._create_vehicles()
#         # Disable collision check for uncontrolled vehicles
#         for vehicle in self.road.vehicles:
#             if vehicle not in self.controlled_vehicles:
#                 vehicle.check_collisions = False
