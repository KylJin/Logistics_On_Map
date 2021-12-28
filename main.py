import json
import argparse
from env import LogisticsEnv
from algo import RandomAgent
import matplotlib.pyplot as plt

AgentDict = {
    'random': RandomAgent,
}


def load_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config


def run_game(args):
    # 读取配置文件
    all_conf = load_config("env/config.json")
    conf = all_conf['Logistics_Transportation']
    map_conf = load_config(f"map/map_{args.map}.json")

    # 根据配置文件创建环境
    env = LogisticsEnv(conf, map_conf)

    # 创建智能体
    n_player = env.n_player
    Agent = AgentDict[args.algo]
    agents = []
    for i in range(n_player):
        action_space = env.get_single_action_space(i)
        agent = Agent(key=i, action_space=action_space)
        agents.append(agent)

    reward_hist = []
    observation = env.reset()
    # 开始仿真
    while not env.is_terminal():
        joint_actions = []
        for idx in range(n_player):
            observation_i = observation[idx]
            action_i = agents[idx].choose_action(observation_i)
            joint_actions.append(action_i)
        observation_, reward, done, info = env.step(joint_actions)

        if not args.silence:
            print("start_storages:", info['start_storages'])
            print("productions:", info['productions'])
            print("demands:", info['demands'])
            print("upper_volume:", info['upper_volume'])
            print()
            print("observation:", observation[0]['obs'])
            print("actual_actions:", info['actual_actions'])
            print("upper_capacity:", info['upper_capacity'])
            print()
            print("reward:", reward)
            print("single_rewards:", info['single_rewards'])
            print('-------------------------------------')

        env.render()
        observation = observation_
        reward_hist.append(reward)

    return reward_hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default=0, type=int)  # 指定地图编号
    parser.add_argument('--algo', default='random', type=str, help="random")
    parser.add_argument('--step_per_update', default=1, type=int)  # 两次更新的step间隔
    parser.add_argument('--silence', action='store_true')
    args = parser.parse_args()

    history = run_game(args)
    print(sum(history))

    # 查看reward趋势
    plt.title(f"Algorithm {args.algo}'s performance on Logistics Env")
    plt.xlabel("day")
    plt.ylabel("reward")
    plt.plot(list(range(len(history))), history)
    plt.show()


if __name__ == '__main__':
    main()
