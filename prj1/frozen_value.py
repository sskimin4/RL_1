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

init(autoreset=True)
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
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
class ValueIteration:
    def __init__(self, env):
        # 환경에 대한 객체 선언
        self.nrow=env.nrow
        self.ncol=env.ncol
        self.env = env
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0.00] *self.nrow for _ in range(self.ncol)]
        # 상 하 좌 우 동일한 확률로 정책 초기화

        # 마침 상태의 설정
        # 감가율
        self.discount_factor = 0.7
        self.possible_actions= [0,1,2,3]
    def value_iteration(self):
        next_value_table = [[0.0] * self.nrow for _ in
                            range(self.ncol)]
        for state in self.get_all_states():
            if state == [self.nrow-1, self.ncol-1]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # 가치 함수를 위한 빈 리스트
            value_list = []

            # 가능한 모든 행동에 대해 계산
            for action in self.possible_actions:
                next_state = self.state_after_action(state, action)
                re = self.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((re + self.discount_factor * next_value))
            # 최댓값을 다음 가치 함수로 대입
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    def get_action(self, state):
        action_list = []
        max_value = -99999
        print(self.value_table)
        if state == [self.nrow-1, self.ncol-1]:
            return []

        # 모든 행동에 대해 큐함수 (보상 + (감가율 * 다음 상태 가치함수))를 계산
        # 최대 큐 함수를 가진 행동(복수일 경우 여러 개)을 반환
        for action in self.possible_actions:
            next_state = self.state_after_action(state, action)
            rew = self.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (rew + self.discount_factor * next_value)
            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)
        return action_list[0]
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
            row = min(row+1, self.nrow-1)
        elif action == 2: # right
            col = min(col + 1, self.ncol-1)
        elif action == 3: # up
            row = max(row - 1, 0)
        return (int(row), int(col))
    def get_reward(self,state, action):
        next_state = self.state_after_action(state, action)
        return reward[next_state[0]][next_state[1]]

value_iteration = ValueIteration(env)
value_iteration.__init__(env)

for _ in range(20):
    value_iteration.value_iteration()
for _ in range(1):
    state_=[0, 0]
    env.reset()
    while True:
        action = value_iteration.get_action(state_)
        print(action)
        state, r, done, info = env.step(action)
        env.render()
        print("State: ", state, "Action: ", action, "Reward: ", r, "Info: ", info)
        state_=value_iteration.state_after_action(state_, action)
        if done:
            print("Finished with reward", r)
            break
