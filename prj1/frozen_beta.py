import gym
from gym.envs.registration import register
from colorama import init
import readchar # pip install readchar
import random
import sys

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}
init(autoreset=True)
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')
env.render()

reward = [[0.00] * env.nrow for _ in range(env.ncol)]
for row in range(env.nrow):
    for col in range(env.ncol):
        if env.desc[row][col] == b'H':
            reward[row][col]=-2.0
        elif env.desc[row][col] == b'G':
            reward[row][col]=4.0
class PolicyIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.nrow=env.nrow
        self.ncol=env.ncol
        self.env = env
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0.00] *self.nrow for _ in range(self.ncol)]
        # 상 하 좌 우 동일한 확률로 정책 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]]*self.nrow for _ in range(self.ncol)]

        # 마침 상태의 설정
        # 감가율
        self.discount_factor = 0.7
        self.possible_actions= [0,1,2,3]
    def policy_evaluation(self):

        # 다음 가치함수 초기화
        next_value_table = [[0.00] * self.nrow
                                    for _ in range(self.ncol)]

        # 모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.get_all_states():
            value = 0.0
            # 마침 상태의 가치 함수 = 0
            if state == [self.nrow-1, self.ncol-1]:
                continue

            # 벨만 기대 방정식
            for action in self.possible_actions:
                next_state = self.state_after_action(state, action)
                re = self.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += self.get_policy(state)[action] * (re + self.discount_factor * next_value)

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # 현재 가치 함수에 대해서 탐욕 정책 발전
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.get_all_states():
            if state == [self.nrow-1, self.ncol-1]:
                continue
            value = -99999
            max_index = []
            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(self.possible_actions):
                next_state = self.state_after_action(state, action)
                rewa = self.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = rewa + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            prob = 1 / len(max_index)

            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동을 반환
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        print(policy)
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # 상태에 따른 정책 반환
    def get_policy(self, state):
        if state == [self.nrow-1, self.ncol-1]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # 가치 함수의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)
    def get_all_states(self):
        all_state=[]
        for x in range(self.nrow):
            for y in range(self.ncol):
                st = [x, y]
                all_state.append(st)
        return all_state
    def state_after_action(self,state, action):
        row = state[0]
        col = state[1]
        if action == 0: #LEFT
            col = max(col - 1, 0)
        elif action == 1: # down
            row = min(row+1, 3)
        elif action == 2: # right
            col = min(col + 1, 3)
        elif action == 3: # up
            row = max(row - 1, 0)
        return (int(row), int(col))
    def get_reward(self,state, action):
        next_state = self.state_after_action(state, action)
        return reward[next_state[0]][next_state[1]]

policy_iteration = PolicyIteration(env)
policy_iteration.__init__(env)

for _ in range(5):
    policy_iteration.policy_evaluation()
    policy_iteration.policy_improvement()
for _ in range(1):
    state_=[0, 0]
    env.reset()
    while True:
        action = policy_iteration.get_action(state_)
        state, r, done, info = env.step(action)
        env.render()
        print("State: ", state, "Action: ", action, "Reward: ", r, "Info: ", info)
        state_=policy_iteration.state_after_action(state_, action)
        if done:
            print("Finished with reward", r)
            break
