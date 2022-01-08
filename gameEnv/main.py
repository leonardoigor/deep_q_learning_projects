from Game import Game
from dueling_ddqn import Agent

env = Game()
agent = Agent(lr=0.0005, gamma=0.99, n_actions=env.n_actions, epsilon=1.0,
              batch_size=64, input_dims=env.n_observation,)

# agent.load_model('./dueling_ddqn/')
for x in range(5000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs)
        obs_, reward, done = env.step(action)
        done = False
        agent.store_transition(obs, action, reward, obs_, done)
        env.draw()
        obs = env.copyArray(obs_)
        score += reward
        if score < -500:
            done = True
        agent.learn()
    agent.save_model('./dueling_ddqn/')
    print(
        f'episode: {x}, score: {score} ep: {agent.epsilon} moves: {env.current_moves}')
