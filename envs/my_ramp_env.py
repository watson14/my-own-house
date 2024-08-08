from typing import Dict, Text, Tuple

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.envs.highway_env import HighwayEnv, Vehicle
from highway_env.envs.common.action import Action


class RampEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics1"
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": np.linspace(10, 30, 20)
            },
            "collision_reward": -200,
            "time_penalty": 1.2,
            "high_speed_reward": 1,
            "safty_distance_reward": -4,
            # "merging_speed_reward": -0.5,
            # "lane_change_reward": -0.05,
            "goal_reward": 4,
            # "initial_lane_id": "b",
            "simulation_frequency": 15,  # [Hz]
            "vehicles_density": 1,
            "screen_width": 1200,
            "screen_height": 600,
            "vehicles_count": 10,
            "policy_frequncy": 2,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        # return utils.lmap(reward,
        #                   [self.config["collision_reward"] + self.config["safty_distance_reward"],
        #                    self.config["high_speed_reward"]  + self.config["goal_reward"]],
        #                   [0, 1])
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        # ttc = self.observation_type.ttc  # 前车车距
        # r_s = 0
        # t = abs(self.observation_type.ttc_to_end_h - self.observation_type.ttc_to_end_l)
        # if ttc <= 1.2:
        #     r_s += np.exp(-ttc / 10)
        # if t <= 1:
        #     r_s += np.exp(-t / 10)

        r_ef = 0
        if self.observation_type.v <= 30 and self.observation_type.v >= 5:
            r_ef = np.clip((self.observation_type.v - 10) / (30 - 10), 0, 1)

        r_m = 0
        if self.observation_type.x > 30 and self.observation_type.x < 300:
            r_m = -np.exp(-self.observation_type.x / 300)
        return {
            "collision_reward": self.vehicle.crashed or self.time > 100,
            # "time_penalty": r_m,
            # "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
            # "high_speed_reward": r_ef,
            # "safty_distance_reward": r_s,
            # "lane_change_reward": action in [0, 2],
            # "merging_speed_reward": sum(  # Altruistic penalty
            #     (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
            #     for vehicle in self.road.vehicles
            #     if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)),
            "goal_reward": bool(self.vehicle.position[0] >= 300 and not self.vehicle.crashed)

        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 300) or self.time > 100

    def _is_truncated(self) -> bool:
        if bool(self.vehicle.position[0] > 300 and not self.vehicle.crashed):
            return True
        return False

    def _reset(self) -> None:
        self.time = 0
        self._make_road()
        self._make_vehicles()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        # obs = self.observation_type.observe()
        # reward = self._reward(action)
        # terminated = self._is_terminated()
        # truncated = self._is_truncated()
        # info = self._info(obs, action)
        self.time += 1
        self._clear_vehicles()
        self._spawn_vehicle()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

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
        net.add_lane("a", "b", StraightLane([0, 0], [150 + 80 , 0], line_types=[c, s]))
        net.add_lane("a", "b", StraightLane([0, 4], [150 + 80, 4], line_types=[s, c]))
        # 与汇入车道结合处
        # net.add_lane("b", "c", StraightLane([150 + 80, -4], [150 + 80 + 80, -4], line_types=[c, c], forbidden=True))
        net.add_lane("b", "c", StraightLane([150 + 80, 0], [150 + 80 + 48, 0], line_types=[c, s], forbidden=True))
        net.add_lane("b", "c", StraightLane([150 + 80, 4], [150 + 80 + 48, 4], line_types=[s, s], forbidden=False))
        # 汇入车道后车道
        # net.add_lane("c", "d", StraightLane([150 + 80 + 80, -4], [150 + 80 + 80 + 150, -4], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([150 + 80 + 48, 0], [150 + 80 + 150, 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([150 + 80 + 48, 4], [150 + 80 + 150, 4], line_types=[s, c]))

        # Merging lane
        amplitude = 3.25
        # ljk = StraightLane([0, 6.5 + 4 + 4], [150, 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        ljk = StraightLane([0, 6 + 4 + 4], [150, 6 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = StraightLane([150, 6 + 4 + 4], [150 + 80, 8], line_types=[c, c], forbidden=True)
        lbc = StraightLane([150 + 80, 8], [150 + 80 + 48, 4], line_types=[n, c], forbidden=True)

        # lkb = SineLane(ljk.position(150, -amplitude), ljk.position(230, -amplitude),
        #                amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        # lbc = SineLane(ljk.position(sum(ends[:2]), -amplitude), ljk.position(sum(ends[:2]), -amplitude - 4),
        #                amplitude, -np.pi / 4, -np.pi / 4, line_types=[c, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "l", lkb)
        net.add_lane("l", "c", lbc)
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
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=15)
        road.vehicles.append(ego_vehicle)
        # ego_vehicle.target_speed = 30

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(5, 0), speed=31.5))

        for _ in range(self.config["vehicles_count"]):
            lane_id = np.random.randint(2)
            lane = road.network.get_lane(("a", "b", lane_id))
            vehicle = other_vehicles_type.create_random(road,
                                                        lane_from="a",
                                                        lane_to="b",
                                                        lane_id=lane_id,
                                                        speed=15 + self.np_random.normal() * 1,#lane.speed_limit,
                                                        spacing=1 / self.config["vehicles_density"],
                                                        ).plan_route_to("d")

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
        lane_id = np.random.randint(2)
        vehicle = vehicle_type.make_on_lane(self.road, ("a" , "b", lane_id),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=15 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
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
                              vehicle in self.controlled_vehicles or vehicle.position[0] < 380 - 4 * vehicle.LENGTH]
    # def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
    #     return "d" in vehicle.lane_index[0] \
    #            and "d" in vehicle.lane_index[1] \
    #            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance



class RampEnv_settledcar(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "goal_reward": 1,
            "initial_lane_id": "b",
            "simulation_frequency": 15,  # [Hz]
            "vehicles_density": 1,
            "screen_width": 600,
            "screen_height": 600,
            "vehicles_count": 10,
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["goal_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1),
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)),
            "goal_reward": bool(self.vehicle.position[0] >= 370)

        }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        if bool(self.vehicle.position[0] >= 370): return True
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

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
        net.add_lane("a", "b", StraightLane([0, -4], [150 + 80, -4], line_types=[c, s]))
        net.add_lane("a", "b", StraightLane([0, 0], [150 + 80, 0], line_types=[s, s]))
        net.add_lane("a", "b", StraightLane([0, 4], [150+80, 4], line_types=[s, c]))
        # 与汇入车道结合处
        net.add_lane("b", "c", StraightLane([150 + 80, -4], [150 + 80 + 80, -4], line_types=[c, c], forbidden=True))
        net.add_lane("b", "c", StraightLane([150 + 80, 0], [150 + 80 + 80, 0], line_types=[c, c], forbidden=True))
        net.add_lane("b", "c", StraightLane([150 + 80, 4], [150 + 80 + 80, 4], line_types=[c, s], forbidden=False))
        # 汇入车道后车道
        net.add_lane("c", "d", StraightLane([150 + 80 + 80, -4], [150 + 80 + 80 + 150, -4], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([150 + 80 + 80, 0], [150 + 80 + 80 + 150, 0], line_types=[s, s]))
        net.add_lane("c", "d", StraightLane([150 + 80 + 80, 4], [150 + 80 + 80 + 150, 4], line_types=[s, c]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
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
                                                     road.network.get_lane(("a", "b", 1)).position(30, 14.5),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=29))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        # road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(5, 0), speed=31.5))

        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle

class my_Exit(HighwayEnv):
    """
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics1",
                #"vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "clip": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [18, 24, 30]
            },
            "lanes_count": 3,
            "collision_reward": -1,
            "high_speed_reward": 0.1,
            "right_lane_reward": 0,
            "normalize_reward": True,
            "goal_reward": 1,
            "vehicles_count": 50,
            "vehicles_density": 1,
            "controlled_vehicles": 1,
            "duration": 18,  # [s],
            "simulation_frequency": 15,
            "scaling": 5
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self._step = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        # obs, reward, terminal, info = super().step(action)
        # info.update({"is_success": self._is_success()})
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

    def _create_road(self, road_length=1500, exit_position=1000, exit_length=100) -> None:
        # net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
        #                                         length=exit_position, nodes_str=("0", "1"))
        # net = RoadNetwork.straight_road_network(self.config["lanes_count"] + 1, start=exit_position,
        #                                         length=exit_length, nodes_str=("1", "2"), net=net)
        # net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=exit_position+exit_length,
        #                                         length=road_length-exit_position-exit_length,
        #                                         nodes_str=("2", "3"), net=net)
        # 直路分成三段
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
                                                length=exit_position, nodes_str=("0", "1"))
        net = RoadNetwork.straight_road_network(self.config["lanes_count"] + 1, start=exit_position,
                                                length=exit_length, nodes_str=("1", "2"), net=net)
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=exit_position+exit_length,
                                                length=road_length-exit_position-exit_length,
                                                nodes_str=("2", "3"), net=net)


        for _from in net.graph:
            for _to in net.graph[_from]:
                for _id in range(len(net.graph[_from][_to])):
                    net.get_lane((_from, _to, _id)).speed_limit = 26 - 3.4 * _id
        exit_position = np.array([exit_position + exit_length, self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH])
        radius = 200
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(center=exit_center,
                            radius=radius,
                            start_phase=3*np.pi/2,
                            end_phase=2*np.pi,
                            forbidden=True)
        net.add_lane("2", "exit", lane)

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            lanes = np.arange(self.config["lanes_count"])
            prob = np.ones(self.config["lanes_count"]) / self.config["lanes_count"]
            lane_id = self.road.np_random.choice(lanes, size=1, p=prob).astype(int)[0]
            vehicle = Vehicle.create_random(self.road,
                                            speed=np.random.normal(30,1),
                                            lane_from="0",
                                            lane_to="1",
                                            lane_id=lane_id,
                                            spacing=self.config["ego_spacing"])
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            # lane_id = self.road.np_random.choice(lanes, size=1,
            #                                      p=lanes / lanes.sum()).astype(int)[0]
            prob = np.ones(self.config["lanes_count"]) / self.config["lanes_count"]
            lane_id = self.road.np_random.choice(lanes, size=1, p=prob).astype(int)[0]
            lane = self.road.network.get_lane(("0", "1", lane_id))

            vehicle = vehicles_type.create_random(self.road,
                                                  lane_from="0",
                                                  lane_to="1",
                                                  lane_id=lane_id,
                                                  speed=lane.speed_limit,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("3")
            # print(lane.speed_limit)
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["goal_reward"]], [0, 1])
            reward = np.clip(reward, 0, 1)
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": self.vehicle.crashed,
            "goal_reward": self._is_success(),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "right_lane_reward": lane_index[-1]
        }

    def _is_success(self):
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        goal_reached = lane_index == ("1", "2", self.config["lanes_count"]) or lane_index == ("2", "exit", 0)
        return goal_reached

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]



# class DenseLidarExitEnv(DenseExitEnv):
#     @classmethod
#     def default_config(cls) -> dict:
#         return dict(super().default_config(),
#                     observation=dict(type="LidarObservation"))
