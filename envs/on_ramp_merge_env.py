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
    n_a = 5
    n_s = 25

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
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "traffic_density": 1,  # easy or hard modes
            "vehicles_count": 3,
            "vehicles_density": 1.2,
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
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, terminated, truncated, info = super().step(action)
        self.time += 1
        self._clear_vehicles()
        self._spawn_vehicle()

        # info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        # info["agents_info"] = agent_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
        # local reward
        # info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        # info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        # obs = np.asarray(obs).reshape((len(obs), -1))
        # obs = obs.reshape(-1)
        return obs, reward, terminated, truncated, info

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 350) #or self.time > 100

    def _is_truncated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        if bool(self.vehicle.position[0] > 350 and not self.vehicle.crashed):
            return True
        return False

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
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=15)
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
            if ab > 0.2:
                road.vehicles.append(
                        other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                            np.random.randint(0, 40) + space, 0), speed=15 + np.random.normal() * 1.5))
            space += 48

        bc = np.random.uniform(0, 1)
        if bc > 0.6:
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("b", "c", 0)).position(
                    np.random.randint(0, 40) + space, 0), speed=15 + np.random.normal() * 1.5))
        cd = np.random.uniform(0, 1)
        if cd > 0.2:
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("c", "d", 0)).position(
                    np.random.randint(0, 40), 0) + space, speed=15 + np.random.normal() * 1.5))
            # vehicle.enable_lane_change = False


        # ramping obstacle car
        if np.random.uniform(0, 1) > 0.8:
            merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                np.random.randint(60, 150), 0), speed=15 + np.random.normal() * 1.5)
            # merging_v.target_speed = 30
            road.vehicles.append(merging_v)

        self.vehicle = ego_vehicle
        # self.controlled_vehicles = []

        # spawn_points_s = [10, 50, 90, 130, 170, 210]
        # spawn_points_m = [5, 45, 85, 125, 165, 205]
        #
        # """Spawn points for CAV"""
        # # spawn point indexes on the straight road
        # spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV //  2, replace=False)
        # # spawn point indexes on the merging road
        # spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
        #                                    replace=False)
        # spawn_point_s_c = list(spawn_point_s_c)
        # spawn_point_m_c = list(spawn_point_m_c)
        # # remove the points to avoid duplicate
        # for a in spawn_point_s_c:
        #     spawn_points_s.remove(a)
        # for b in spawn_point_m_c:
        #     spawn_points_m.remove(b)
        #
        # """Spawn points for HDV"""
        # # spawn point indexes on the straight road
        # spawn_point_s_h = np.random.choice(spawn_points_s, 2, replace=False)
        # # spawn point indexes on the merging road
        # spawn_point_m_h = np.random.choice(spawn_points_m, 2,
        #                                    replace=False)
        # spawn_point_s_h = list(spawn_point_s_h)
        # spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        # initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        # loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        # initial_speed = list(initial_speed)
        # loc_noise = list(loc_noise)
        #
        # """spawn the CAV on the straight road first"""
        # for _ in range(num_CAV // 2):
        #     ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
        #         spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
        #     self.controlled_vehicles.append(ego_vehicle)
        #     road.vehicles.append(ego_vehicle)
        # """spawn the rest CAV on the merging road"""
        # for _ in range(num_CAV - num_CAV // 2):
        #     ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
        #         spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
        #     self.controlled_vehicles.append(ego_vehicle)
        #     road.vehicles.append(ego_vehicle)
        #
        # """spawn the HDV on the main road first"""
        # for _ in range(1):
        #     road.vehicles.append(
        #         other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
        #             # spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
        #             # speed=initial_speed.pop(0)))
        #             150, 0), speed=15))

        #
        # """spawn the rest HDV on the merging road"""
        # for _ in range(num_HDV - num_HDV // 2):
        #     road.vehicles.append(
        #         other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
        #             spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
        #                             speed=initial_speed.pop(0)))

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
        # vehicle.randomize_behavior()
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
