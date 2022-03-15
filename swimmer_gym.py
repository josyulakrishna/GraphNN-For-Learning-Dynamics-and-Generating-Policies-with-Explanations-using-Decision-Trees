import gym
import numpy as np
env = gym.make("Swimmer-v2")
done = False
env.reset()
while not done:
    action = np.random.uniform(low=-1,high=1, size=2)
    obs, reward, done, info = env.step(action)
    print(obs.size)
    # env.render()