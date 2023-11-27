# encoding=utf-8
import gym
import numpy as np

class RL_DynamicProgramming:  # 基于动态规划的策略迭代和价值迭代算法实现
    def __init__(self, env, tol=1e-6, gamma=1.0, is_policy=True):
        self.env       = env                                  # 环境对象
        self.states    = env.observation_space.n              # 状态数量（离散状态）
        self.actions   = env.action_space.n                   # 动作数量（离散动作）
        self.P         = self.env.P                           # 环境的动力系统
        self.tol       = tol                                  # 误差阈值
        self.gamma     = gamma                                # 缩减因子γ
        self.is_policy = is_policy                            # 策略迭代或价值迭代
        self.v         = np.zeros(self.states)                # 状态价值v(s)
        self.policy    = np.ones((self.states, self.actions)) # 行动策略policy(s,a)
        self.policy    = self.policy / self.actions           # 随机策略（相同概率）

    def v2q(self, s):  # 对一个状态，基于状态价值计算动作价值
        q_s = np.zeros(self.actions)                          # 用于存储q(s,a)
        for a in range(self.actions):                         # 遍历所有动作
            for prob, s_, reward, done in self.P[s][a]:       # 遍历所有可能结果
                q_s[a] += prob * reward                       # 累加收益r(s,a)
                q_s[a] += prob * self.gamma * self.v[s_]      # 累加p(s'|s,a)*γ*v(s')

        return q_s                                            # 返回状态的动作价值

    def evaluate_policy(self):  # 策略评估
        while True:                                           # 循环直到收敛
            delta = 0.0                                       # 用于存储状态价值变化量
            for s in range(self.states):                      # 遍历所有状态
                q_s   = self.v2q(s)                           # 计算动作价值q(s,a)
                v_s   = np.sum(self.policy[s] * q_s)          # 计算新的状态价值v(s)
                delta = max(delta, abs(self.v[s]-v_s))        # 计算状态价值变化量
                self.v[s] = v_s                               # 更新状态价值v(s)
            if delta < self.tol:                              # 如果最大变化量小于阈值
                break                                         # 中止循环

    def improve_policy(self):  # 策略改进
        is_changed = False                                    # 用于判断策略是否变化
        for s in range(self.states):                          # 遍历所有状态
            q_s         = self.v2q(s)                         # 计算动作价值q(s,a)
            q_s_max     = np.max(q_s)                         # 求动作价值最大值
            action_list = np.where(q_s == q_s_max)[0]         # 获取最大值的下标
            policy_s    = np.zeros(self.actions)              # 用于存储状态s的策略
            p           = 1.0 / len(action_list)              # 所有最优动作的概率相同
            policy_s[action_list] = p                         # 为状态s的动作策略赋值

            if not (self.policy[s] == policy_s).all():        # 状态s的策略是否变化
                is_changed     = True                         # 设置策略改进标志
                self.policy[s] = policy_s                     # 更新状态s的行动策略

        return is_changed                                     # 返回策略是否变化标志

    def iterate_policy(self):  # 策略迭代
        while True:                                           # 循环直到策略迭代收敛
            self.evaluate_policy()                            # 策略评估
            changed = self.improve_policy()                   # 策略改进
            if not changed:                                   # 策略是否不再变化
                break                                         # 中止循环

    def iterate_value(self):  # 价值迭代
        while True:                                           # 循环直到价值迭代收敛
            delta = 0.0                                       # 用于存储状态价值变化量
            for s in range(self.states):                      # 遍历所有状态
                q_s   = self.v2q(s)                           # 计算状态s的所有动作价值
                v_s   = np.max(q_s)                           # 计算动作价值最大值
                delta = max(delta, abs(self.v[s]-v_s))        # 计算状态价值变化量
                self.v[s] = v_s                               # 更新状态价值v(s)
            if delta < self.tol:                              # 如果最大变化量小于阈值
                break                                         # 中止循环

        self.improve_policy()                                 # 输出最优策略

def main():
    np.set_printoptions(precision=4)                          # 设置保留4位小数
    env = gym.make("FrozenLake-v1")                           # 生成冰面滑行环境对象
    print(env.observation_space)                              # 打印状态空间
    print(env.action_space, "\n")                             # 打印动作空间
    print(env.P[14][1])                                       # 打印状态14动作1的结果
    print(env.P[15], "\n")                                    # 打印状态15所有动作结果

    agent1 = RL_DynamicProgramming(env, is_policy=True)       # 生成策略迭代智能体对象
    agent1.iterate_policy()                                   # 执行策略迭代
    print(agent1.v.reshape(4, 4))                             # 打印最优状态价值
    print(agent1.policy, "\n")                                # 打印最优行动策略

    agent2 = RL_DynamicProgramming(env, is_policy=False)      # 生成价值迭代智能体对象
    agent2.iterate_value()                                    # 执行价值迭代
    print(agent2.v.reshape(4, 4))                             # 打印最优状态价值
    print(agent2.policy)                                      # 打印最优行动策略

if __name__ == "__main__":
    main()