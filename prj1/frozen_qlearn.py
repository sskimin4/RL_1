import gym
from gym.envs.registration import register
from colorama import init
import numpy as np
import readchar # pip install readchar
import random
import sys
from collections import defaultdict

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

init(autoreset=True)
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True}
)

env = gym.make('FrozenLake-v3')

class QLearningAgent:
    def __init__(self, actions):
        # 환경에 대한 객체 선언
        self.actions = actions
        self.learning_rate = 0.45
        self.discount_factor = 0.9
        self.epsilon = 0.99
        self.q_table = [[0.00] * 4 for _ in range((env.nrow*env.ncol))]

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        #print(next_state)
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)
    def get_action(self, state):
        if np.random.rand() > self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
nrow=env.nrow
ncol=env.ncol

agent = QLearningAgent(actions=[0,1,2,3])
episode = 50000
for _ in range(episode):
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, r, done, info = env.step(action)
        #env.render()
        #print("State: ", state, "Action: ", action, "Reward: ", r, "Info: ", info)
        agent.learn(state,action,r,next_state)
        state=next_state
        if done:
            #print("Finished with reward", r)
            break
state= env.reset()
while True:
    action = agent.get_action(state)
    next_state, r, done, info = env.step(action)
    env.render()
    print("State: ", state, "Action: ", action, "Reward: ", r, "Info: ", info)
    agent.learn(state,action,r,next_state)
    state=next_state
    if done:
        print("Finished with reward", r)
        break
