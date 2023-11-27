# encoding=utf-8
import gym  # 导入Gym库

def main():
    # 获取和打印Gym库所有环境ID
    env_ids = gym.envs.registry.keys()   # 获取Gym库所有环境的ID
    for env_id in env_ids:               # 遍历Gym库所有环境的ID
        print(env_id)                    # 打印Gym库的每个环境ID

    # 使用make函数生成环境对象
    env = gym.make('FrozenLake-v1', render_mode='human')      # 使用冰湖滑行环境
    # env = gym.make('CliffWalking-v0', render_mode='human')  # 使用悬崖行走环境
    # env = gym.make("Blackjack-v1", render_mode='human')     # 使用二十一点环境
    # env = gym.make('CartPole-v1', render_mode='human')      # 使用车杆平衡环境

    # 打印环境的状态空间和动作空间
    print("\nstate and action space:")
    print(env.observation_space)
    print(env.action_space)

    # 每个环境执行10个回合（使用随机策略）
    for i_episode in range(10):
        # 使用reset函数重置环境，返回初始状态和提示信息
        observation, info = env.reset()
        # 打印初始状态和提示信息
        print("\ni_episode = %d, initial state info:" % i_episode)
        print(observation, info)
        # 在每个回合，随机走10步
        for t in range(10):
            # 执行env.reset()或env.step()之后，使用render函数显示图形化的当前环境
            env.render()
            # 在当前环境的动作空间中随机选择一个动作，并打印动作
            action = env.action_space.sample()
            print("action = ", action)
            # 在当前环境中执行一个动作，并打印返回信息
            observation, reward, terminated, truncated, info = env.step(action)
            print("info   = ", observation, reward, terminated, truncated, info)
            # 如果到达终止状态或被中断，打印总共走了多少步
            if terminated or truncated:
                print("Episode finished after %d steps\n" % (t+1))
                break
    # 使用完环境后关闭环境
    env.close()

if __name__ == "__main__":
    main()