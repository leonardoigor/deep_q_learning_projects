from Game import Game

env = Game()
for x in range(50):
    obs = env.reset()
    done = False

    while not done:
        obs_, reward, done = env.step([1, 0, 0, 1])
        done = False
        print(reward)

        env.draw()
    print(f'episode: {x}')
