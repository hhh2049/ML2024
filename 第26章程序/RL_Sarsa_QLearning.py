# encoding=utf-8
import gym
import numpy as np

class Sarsa_QLearning_Agent:  # Sarsa和Q学习算法的实现
    def __init__(self, env, epsilon=0.05, eta=0.5, gamma=0.95, n_max=2000, is_sarsa=True):
        self.env      = env                                    # 环境对象
        self.states   = env.observation_space.n                # 状态数量
        self.actions  = env.action_space.n                     # 动作数量
        self.epsilon  = epsilon                                # ε-贪婪策略的探索率ε
        self.eta      = eta                                    # 学习率η
        self.gamma    = gamma                                  # 折扣因子γ
        self.n_max    = n_max                                  # 训练的回合数
        self.is_sarsa = is_sarsa                               # sarsa还是Q学习算法
        self.q_table  = np.zeros((self.states, self.actions))  # 定义Q表格

    def sample(self, state, is_train=True):  # 根据当前状态，选择一个动作
        if is_train:                                           # 训练时使用ε-贪婪策略
            if np.random.uniform(0, 1) <= self.epsilon:        # 如果随机数小于等于ε
                action = np.random.choice(self.actions)        # 随机选择动作（探索）
            else:                                              # 如果随机数大于ε
                action = self.predict(state)                   # 选择最优动作（利用）
        else:                                                  # 如果不是训练
            action = self.predict(state)                       # 选择最优动作

        return action                                          # 返回选中的动作

    def predict(self, state):  # 根据当前状态，选择最优动作
        q_values    = self.q_table[state, :]                   # 根据状态选中Q表格某行
        max_q       = np.max(q_values)                         # 获取该行的最大值
        action_list = np.where(q_values == max_q)[0]           # 获取最大值的下标
        action      = np.random.choice(action_list)            # 任选一个最优动作

        return action                                          # 返回选中的动作

    def learn(self, s, a, r, s_, a_, done):  # 实现Sarsa和Q学习算法
        eta, gamma = self.eta, self.gamma                      # 为缩短后面代码长度
        q_predict  = self.q_table[s, a]                        # 获取预测值q(s,a)
        if not done:                                           # 如果不是终止状态
            if self.is_sarsa:                                  # 如果是Sarsa学习算法
                q_target = r + gamma * self.q_table[s_, a_]    # 计算目标值q(s,a)
            else:                                              # 如果是Q学习算法
                q_target = r + gamma * self.q_table[s_].max()  # 计算目标值q(s,a)
        else:                                                  # 如果是终止状态
            q_target = r                                       # 计算目标值q(s,a)

        self.q_table[s, a] += eta * (q_target - q_predict)     # 更新Q表格

    def play_episode(self, is_train=True):  # 玩一个回合
        total_reward = 0                                       # 用于存储总的收益
        state, _     = self.env.reset()                        # 重置环境
        if not is_train: print("\n%d" % state, end=" ")        # 未训练时打印初始状态
        action       = self.sample(state, is_train)            # 选择一个动作
        while True:                                            # 循环直到回合结束
            state_, reward, done, _, _ = self.env.step(action) # 走一步
            action_ = self.sample(state_, is_train)            # 根据新状态选择动作
            if is_train:                                       # 如果是在训练
                s, a, r = state, action, reward                # 为缩短代码长度
                s_, a_  = state_, action_                      # 为缩短代码长度
                self.learn(s, a, r, s_, a_, done)              # 执行学习算法
            else:                                              # 如果没有在训练
                print(state_, end=" ")                         # 打印最终训练结果
            state, action = state_, action_                    # 更新状态和动作
            total_reward += reward                             # 累加奖励
            if done:                                           # 如果已是终止状态
                break                                          # 终止循环

        return total_reward                                    # 返回本回合收益

    def train(self):  # 执行训练任务
        for episode in range(self.n_max):                      # 循环直到达到回合数
            total_reward = self.play_episode()                 # 执行一个回合的训练
            if episode % 100 == 0:                             # 每隔100个回合
                print("Total Reward =", total_reward)          # 打印一次训练得分

        for i in range(self.states):                           # 遍历所有状态
            print("{} : {}".format(i, self.q_table[i, :]))     # 逐行打印Q表格

    def print_environment(self):
        env = self.env                                         # 为缩短代码长度
        print("观测空间 = {}".format(env.observation_space))   # 打印观测空间
        print("动作空间 = {}".format(env.action_space))        # 打印动作空间
        print("状态数量 = {}".format(env.nS))                  # 打印状态数量
        print("动作数量 = {}".format(env.nA))                  # 打印动作数量
        print("地图形状 = {}\n".format(env.shape))             # 打印地图形状

def main():
    env   = gym.make("CliffWalking-v0")                        # 生成环境对象
    agent = Sarsa_QLearning_Agent(env, is_sarsa=True)          # 生成智能体对象
    agent.print_environment()                                  # 打印环境信息
    agent.train()                                              # 执行训练
    agent.play_episode(is_train=False)                         # 打印训练结果

    agent = Sarsa_QLearning_Agent(env, is_sarsa=False)         # 生成智能体对象
    agent.train()                                              # 执行训练
    agent.play_episode(is_train=False)                         # 打印训练结果

if __name__ == "__main__":
    main()
