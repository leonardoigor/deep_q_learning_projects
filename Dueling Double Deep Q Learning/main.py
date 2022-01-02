

import gym
import numpy as np
from dueling_ddqn import Agent
import json


env = gym.make('LunarLander-v2')
agent = Agent(lr=0.0005, gamma=0.99, n_actions=4, epsilon=1.0,
              batch_size=64, input_dims=[8])

n_games = 2000
scores = []
eps_history = []
avg_scores = []


def toJSon():
    data = {
        'scores': scores,
        'eps_history': eps_history,
        'avg_scores': avg_scores
    }
    with open('./ddqn.json', 'w') as f:
        json.dump(data, f)


agent.load_model('./models/lunar/lunar_ddqn')
for i in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        # env.render()
        agent.learn()
    eps_history.append(agent.epsilon)
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    print('episode ', i, 'score %.2f' % score,
          'average score %.2f' % avg_score,
          'epsilon %.2f' % agent.epsilon)
    agent.save_model('./models/lunar/lunar_ddqn')
    toJSon()
