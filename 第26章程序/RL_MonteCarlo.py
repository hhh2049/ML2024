# encoding=utf-8
import gym
import numpy as np

class Monte_Carlo_Agent:  # 蒙特卡洛强化学习算法实现
    def __init__(self, env, e_greedy=0.1, gamma=1.0, n_max=50000):
        self.env     = env                                      # 环境对象
        self.epsilon = e_greedy                                 # ε-贪婪策略的探索率ε
        self.gamma   = gamma                                    # 折扣因子γ
        self.n_max   = n_max                                    # 指定训练的回合总数
        self.actions = env.action_space.n                       # 动作数量（离散动作）
        self.q_table = {}                                       # 用于存储Q表格q(s,a)
        self.c_table = {}                                       # 用于存储计数器c(s,a)

    def predict(self, state):  # 根据当前状态，选择最优动作
        q_values = np.array(list(self.q_table[state].values())) # 获取q(s,a)的所有值
        q_max    = np.max(q_values)                             # 获取q(s,a)的最大值
        actions  = np.where(q_values == q_max)[0]               # 获取最大值的下标
        action   = np.random.choice(actions)                    # 任选一个最优动作

        return action                                           # 返回选中的最优动作

    def sample(self, state, is_train=True):  # 根据当前状态和ε-贪婪策略，选择一个动作
        if is_train:                                            # 训练时使用ε-贪婪策略
            if np.random.uniform(0, 1) <= self.epsilon:         # 如果随机数小于等于ε
                action = np.random.choice(self.actions)         # 随机选择动作（探索）
            else:                                               # 如果随机数大于ε
                action = self.predict(state)                    # 选择最优动作（利用）
        else:                                                   # 如果不是训练
            action = self.predict(state)                        # 直接选择最优动作

        return action                                           # 返回选中的动作

    def sample_episode(self):  # 玩一个回合
        episode  = []                                           # 存放状态动作奖励序列
        state, _ = self.env.reset()                             # 重置环境
        while True:                                             # 循环直到一个回合结束
            if state not in self.q_table.keys():                # 如果首次遇到该状态
                self.q_table[state] = {0: 0.0, 1: 0.0}          # 为该状态定义q(s,a)
                self.c_table[state] = {0: 0.0, 1: 0.0}          # 为该状态定义c(s,a)

            action = self.sample(state)                         # 选择一个动作
            state_, reward, done, _, _ = self.env.step(action)  # 走一步
            episode.append((state, action, reward))             # 保存状态动作奖励
            if done:                                            # 如果已达终止状态
                break                                           # 中止循环
            state = state_                                      # 更新状态
        return episode                                          # 返回状态动作奖励

    def train(self):  # 执行训练任务
        for i in range(self.n_max):                             # 执行每个回合的训练
            if i % 1000 == 0:                                   # 每隔1000个回合
                print("episode i = {}".format(i))               # 打印一次进度信息

            episode = self.sample_episode()                     # 根据策略玩一个回合
            G, T    = 0.0, len(episode)                         # 初始化收益、步骤数
            for t in range(T-1, -1, -1):                        # 从T-1到0，循环T次
                state, action, reward = episode[t]              # 获取状态动作奖励
                G = reward + self.gamma * G                     # 计算G

                self.c_table[state][action] += 1                # q(s,a)的计数加1
                q_sa = self.q_table[state][action]              # 为缩短代码长度
                c_sa = self.c_table[state][action]              # 为缩短代码长度
                self.q_table[state][action] += (G - q_sa)/c_sa  # 更新Q表格

        print("\nQ Table length =", len(self.q_table))          # 打印Q表格的长度
        # print("Q Table:\n", self.q_table)                     # 打印Q表格

    def play_episode_with_policy(self, random=True):  # 比较随机策略与最优策略的收益
        env          = self.env                                 # 用于缩短代码长度
        total_reward = 0                                        # 用于存储总的收益
        state, _     = env.reset()                              # 重置环境
        while True:                                             # 循环直到回合结束
            if random:                                          # 如果按随机策略玩
                action = np.random.choice(env.action_space.n)   # 随机选择动作
            else:                                               # 如果按最优策略玩
                action = self.sample(state, is_train=False)     # 选择最优动作
            state, reward, terminated, _, _ = env.step(action)  # 走一步
            total_reward += reward                              # 累积奖励
            if terminated:                                      # 如果已达终止状态
                break                                           # 中止循环
        return total_reward                                     # 返回总的收益

    def test(self, times=10000, random=True):  # 测试随机策略和最优策略的平均收益
        rewards = 0.0                                           # 用于存储总的收益
        for i in range(times):                                  # 玩times个回合
            rewards += self.play_episode_with_policy(random)    # 玩一个回合并累积收益
        if random:                                              # 如果是随机策略
            print("random policy reward =", rewards / times)    # 打印随机策略平均收益
        else:                                                   # 如果是最优策略
            print("best   policy reward =", rewards / times)    # 打印最优策略平均收益

    def play_episode_for_print(self):  # 玩一个回合
        total_reward = 0
        env          = self.env
        state, _     = env.reset()
        print('初始状态={}'.format(state))

        while True:
            print('玩家的牌={}, 庄家的牌={}'.format(env.player, env.dealer))
            action = np.random.choice(env.action_space.n)
            print('动作={}'.format(action))
            state, reward, terminated, truncated, info = env.step(action)
            print('状态={}, 奖励={}, 结束标志={}, 截断标志={}, 提示信息={}'.format(
                state, reward, terminated, truncated, info))
            total_reward += reward
            if terminated or truncated:
                break
        print('玩家终牌={}, 庄家终牌={}'.format(env.player, env.dealer))
        print('总的收益={}\n'.format(total_reward))

def main():
    env = gym.make("Blackjack-v1")                              # 生成环境对象
    print("state  space = {}".format(env.observation_space))    # 打印观测空间
    print("action space = {}\n".format(env.action_space))       # 打印动作空间
    agent = Monte_Carlo_Agent(env)                              # 生成智能体对象
    agent.play_episode_for_print()                              # 打印一个回合信息
    agent.train()                                               # 执行训练
    agent.test(times=10000, random=True)                        # 打印随机策略收益
    agent.test(times=10000, random=False)                       # 打印最优策略收益
    env.close()

if __name__ == "__main__":
    main()
