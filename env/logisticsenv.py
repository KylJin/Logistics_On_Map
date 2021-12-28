from env.simulators.game import Game
from env.obs_interfaces.observation import DictObservation
from env.rendering.viewer import Viewer
import numpy as np
from utils.box import Box
import random


class LogisticsEnv(Game, DictObservation):
    def __init__(self, conf, map_conf):
        super().__init__(map_conf['n_vertex'], conf['is_obs_continuous'], conf['is_act_continuous'],
                         conf['game_name'], map_conf['n_vertex'], conf['obs_type'])
        self.map_conf = map_conf
        self.n_goods = int(map_conf['n_goods'])
        self.max_step = int(conf['max_step'])

        self.players = []
        self.init_map()  # 根据地图数据初始化地图

        self.step_cnt = 0
        self.n_return = [0] * self.n_player
        self.current_state = None
        self.all_observes = None
        # 每个玩家的action space list, 可以根据player_id获取对应的single_action_space
        self.joint_action_space = self.set_action_space()
        self.info = {}

        self.viewer = None

    def init_map(self):
        goods_list = []
        for goods_info in self.map_conf['goods']:
            goods = Goods(goods_info)
            goods_list.append(goods)

        # 添加图中的节点
        vertices = self.map_conf['vertices'].copy()
        for vertex_info in vertices:
            vertex_info.update({'goods': goods_list})
            vertex = Vertex(vertex_info['key'], vertex_info)
            self.players.append(vertex)

        # 添加图中的边
        roads = self.map_conf['roads'].copy()
        for road_info in roads:
            start = road_info['start']
            end = road_info['end']
            road = Road(road_info)
            self.players[start].add_neighbor(end, road)
            if not self.map_conf['is_graph_directed']:  # 若是无向图，则加上反方向的边
                self.players[end].add_neighbor(start, road)

        # 初始化每个节点
        for i in range(self.n_player):
            self.players[i].update_init_storage()

    def reset(self):
        self.players = []
        self.init_map()

        self.step_cnt = 0
        self.n_return = [0] * self.n_player

        self.current_state = self.get_current_state()
        self.all_observes = self.get_all_observations()

        self.info = {
            'productions': [self.players[i].production for i in range(self.n_player)],
            'upper_volume': [self.players[i].upper_capacity for i in range(self.n_player)],
            'upper_capacity': [[act.high.tolist() for act in v_action] for v_action in self.joint_action_space]
        }

        return self.all_observes

    def step(self, all_actions):
        self.step_cnt += 1
        all_actions = self.bound_actions(all_actions)

        self.current_state = self.get_next_state(all_actions)
        self.all_observes = self.get_all_observations()

        reward, single_rewards = self.get_reward(all_actions)
        done = self.is_terminal()
        self.info.update({
            'actual_actions': all_actions,
            'single_rewards': single_rewards
        })

        return self.all_observes, reward, done, self.info

    def bound_actions(self, all_actions):  # 对每个节点的动作进行约束
        bounded_actions = []
        for i in range(self.n_player):
            result = self.players[i].bound_actions(all_actions[i])
            bounded_actions.append(result)

        return bounded_actions

    def get_current_state(self):
        current_state = []
        for i in range(self.n_player):
            current_state.append(self.players[i].init_storage.copy())

        return current_state

    def get_next_state(self, all_actions):
        assert len(all_actions) == self.n_player

        # 统计每个节点当天运出的货物量out_storages，以及接收的货物量in_storages
        out_storages = np.zeros((self.n_player, self.n_goods))
        in_storages = np.zeros((self.n_player, self.n_goods))
        for i in range(self.n_player):
            action = np.array(all_actions[i])
            out_storages[i] = np.sum(action, axis=0)
            connections = self.players[i].get_connections()
            for idx, nbr in enumerate(connections):
                in_storages[nbr] += action[idx]

        # 更新每个节点当天的最终库存量以及下一天的初始库存量，
        # 并记录每个节点当天最开始的初始库存start_storages和消耗量demands，用于可视化
        next_state = []
        start_storages, demands = [], []
        for i in range(self.n_player):
            start_storages.append(self.players[i].final_storage.copy())
            demands.append(self.players[i].demands)

            self.players[i].update_final_storage(out_storages[i], in_storages[i])
            self.players[i].update_init_storage()
            next_state.append(self.players[i].init_storage.copy())

        self.info.update({
            'start_storages': start_storages,
            'demands': demands
        })

        return next_state

    def get_dict_observation(self, current_state, player_id, info_before):
        obs = {
            "obs": current_state,
            "connected_player_index": self.players[player_id].get_connections(),
            "controlled_player_index": player_id
        }
        return obs

    def get_all_observations(self, info_before=''):
        all_obs = self.get_dict_many_observation(
            self.current_state,
            range(self.n_player),
            info_before
        )
        return all_obs

    def get_reward(self, all_actions):
        total_reward = 0
        single_rewards = []
        for i in range(self.n_player):
            action = all_actions[i]
            reward = self.players[i].calc_reward(action)
            total_reward += reward
            single_rewards.append(reward)
            self.n_return[i] += reward

        return total_reward, single_rewards

    def set_action_space(self):
        action_space = []

        for i in range(self.n_player):
            vertex = self.players[i]
            action_space_i = []
            for j in vertex.get_connections():
                road = vertex.get_road(j)
                high = [road.upper_capacity // vertex.goods[k].volume for k in range(self.n_goods)]
                space = Box(np.zeros(self.n_goods), np.array(high), dtype=np.float64)
                action_space_i.append(space)
            action_space.append(action_space_i)

        return action_space

    def get_single_action_space(self, player_id):
        return self.joint_action_space[player_id]

    def is_terminal(self):
        is_done = self.step_cnt >= self.max_step
        return is_done

    def get_render_data(self, current_state=None):
        render_data = {
            'day': self.step_cnt,
            'storages': self.info['start_storages'],
            'productions': self.info['productions'],
            'demands': self.info['demands'],
            'total_reward': sum(self.n_return),
            'single_rewards': self.info['single_rewards'],
            'actions': self.info['actual_actions']
        }

        return render_data

    def render(self):
        if self.viewer is None:
            pd_gaps, all_connections, all_times = [], [], []
            for i in range(self.n_player):
                vertex = self.players[i]
                pd_gaps.append(sum([p - d for p, d in zip(vertex.production, vertex.lambda_)]))
                connections = vertex.get_connections()
                all_connections.append(connections)
                times = [vertex.get_road(j).trans_time for j in connections]
                all_times.append(times)
            network_data = {
                'n_vertex': self.n_player,
                'is_abs_coords': self.map_conf.get('is_abs_coords'),
                'v_coords': self.map_conf.get('coords'),
                'connections': all_connections,
                'pd_gaps': pd_gaps,
                'trans_times': all_times
            }
            self.viewer = Viewer(1200, 800, network_data)

        render_data = self.get_render_data()
        self.viewer.render(render_data)


class Vertex(object):
    def __init__(self, key, info):
        self.key = key
        self.connectedTo = {}

        self.goods = info['goods']
        self.n_goods = len(self.goods)
        self.production = info['production']
        self.init_storage = [0] * self.n_goods
        self.final_storage = info['init_storage']
        self.upper_capacity = info['upper_capacity']

        self.store_cost = info['store_cost']
        self.loss_cost = info['loss_cost']
        self.storage_loss = [0] * self.n_goods  # 更新完当天的最终库存量后，统计当天的库存溢出量
        self.init_storage_loss = [0] * self.n_goods  # 因为每次状态更新会提前计算下一天的初始库存量，
        # 若不单独记录初始库存的溢出量，则会在计算每日reward时出错
        self.lambda_ = info['lambda']

        self.demands = None
        self.fulfillment = None

    def add_neighbor(self, nbr, road):
        self.connectedTo.update({nbr: road})

    def get_connections(self):
        return list(self.connectedTo.keys())

    def get_road(self, nbr):
        return self.connectedTo.get(nbr)

    def get_demands(self):
        demands = [np.random.poisson(lam=self.lambda_[k], size=1)[0] for k in range(self.n_goods)]
        return demands

    def bound_actions(self, actions):
        if len(actions) == 0:  # 若该节点没有动作，则不做约束
            return actions

        actions = np.array(actions)
        result = np.zeros_like(actions)
        for k in range(self.n_goods):
            actual_trans = np.sum(actions[:, k])
            if self.init_storage[k] > 0:
                if actual_trans > self.init_storage[k]:  # 运出的总货物量超过初始库存量
                    # 所有道路上同种货物量进行等比例缩放
                    result[:, k] = actions[:, k] * self.init_storage[k] / actual_trans
                else:  # 合法动作
                    result[:, k] = actions[:, k]

        goods_volumes = np.array([goods.volume for goods in self.goods])
        total_volumes = np.dot(result, goods_volumes.T)  # 每条道路上运输的总体积
        for i, nbr in enumerate(self.get_connections()):
            road = self.get_road(nbr)
            if total_volumes[i] > road.upper_capacity:  # 该道路上货物总体积超过道路容量
                # 对该道路上每种货物，按其体积占比，对数量进行等比例缩放
                result[i] = result[i] * road.upper_capacity / total_volumes[i]

        return result.tolist()

    def update_init_storage(self):
        self.demands = self.get_demands()
        total_volume = 0
        for k in range(self.n_goods):
            self.init_storage[k] = self.final_storage[k] - self.demands[k] + self.production[k]
            if self.init_storage[k] > 0:
                total_volume += self.init_storage[k] * self.goods[k].volume

        self.init_storage_loss = [0] * self.n_goods
        while total_volume > self.upper_capacity:  # 当天初始库存超过存储容量上限
            key = random.randint(0, self.n_goods - 1)
            if self.init_storage[key] > 0:
                self.init_storage[key] -= 1
                self.init_storage_loss[key] += 1
                total_volume -= self.goods[key].volume

    def update_final_storage(self, out_storage, in_storage):
        self.fulfillment = self.demands.copy()
        total_volume = 0
        for k in range(self.n_goods):
            self.fulfillment[k] -= min(0, self.final_storage[k])
            self.final_storage[k] = self.init_storage[k] - out_storage[k] + in_storage[k]
            if self.final_storage[k] > 0:
                total_volume += self.final_storage[k] * self.goods[k].volume

        self.storage_loss = self.init_storage_loss
        while total_volume > self.upper_capacity:  # 当天最终库存超过存储容量上限
            key = random.randint(0, self.n_goods - 1)
            if self.final_storage[key] > 0:
                self.final_storage[key] -= 1
                self.storage_loss[key] += 1
                total_volume -= self.goods[key].volume

        for k in range(self.n_goods):
            self.fulfillment[k] += min(0, self.final_storage[k])

    def calc_reward(self, actions, mu=1, scale=100):
        connections = self.get_connections()
        assert len(actions) == len(connections)

        reward = 0
        for k in range(self.n_goods):
            goods = self.goods[k]
            # 1. 需求满足回报
            reward += goods.price * self.fulfillment[k]
            # 2. 货物存储成本（兼惩罚项）
            if self.final_storage[k] >= 0:
                reward -= (self.store_cost * self.final_storage[k] * goods.volume)
            else:
                reward += (mu * self.final_storage[k] * goods.volume)
            # 3. 舍弃货物损失
            reward -= self.loss_cost[k] * self.storage_loss[k]

        # 4. 货物运输成本
        for (action, nbr) in zip(actions, connections):
            road = self.get_road(nbr)
            volume = 0
            for k in range(self.n_goods):
                volume += action[k] * self.goods[k].volume
            reward -= (road.trans_cost * road.trans_time * volume)

        return reward / scale


class Road(object):
    def __init__(self, info):
        self.upper_capacity = info['upper_capacity']
        self.trans_time = info['trans_time']
        self.trans_cost = info['trans_cost']


class Goods(object):
    def __init__(self, info):
        self.volume = info['volume']
        self.price = info['price']
