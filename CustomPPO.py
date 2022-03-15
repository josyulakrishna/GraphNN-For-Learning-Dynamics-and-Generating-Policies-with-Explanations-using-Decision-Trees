import sys

import dm_control.suite
from dm_control import suite
from dm_control import viewer
import numpy as np
# from dm_control.suite.swimmer import swimmer
import dmc2gym
from stable_baselines3 import PPO, TD3

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines import TRPO
# from stable_baselines import TD3
# from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


# from stable_baselines3 import common.noise.NormalActionNoise
import pdb
##################
#### visualize ###
##################

# env = suite.load(domain_name="myswimmer", task_name="swimmer")
# # env = suite.load(domain_name="walker", task_name="run")
#
# action_spec = env.action_spec()
# # print(action_spec)
# obs_spec = env.observation_spec()
# # print(obs_spec)
# print("observation joints",  obs_spec["joints"].shape)
# print("observation bv",  obs_spec["body_velocities"].shape)
# print("observation abs",  obs_spec["abs"].shape)

# sys.exit(0)
# # Define a uniform random policy.
# # action = np.random.uniform(low=action_spec.minimum,
# #                            high=action_spec.maximum,
# #                            size=action_spec.shape)
# # for i in range(10):
# #   env.step(action)

# def random_policy(time_step):
#   del time_step  # Unused.
#   # return np.random.uniform(low=action_spec.minimum,
#   #                          high=action_spec.maximum,
#   #                          size=action_spec.shape)
#   return np.random.uniform(low=-1,
#                            high=1,
#                            size=action_spec.shape)
# viewer.launch(env, policy=random_policy)

####################

#####################
### test gym env ####
# #
# env = dmc2gym.make(domain_name='myswimmer', task_name='swimmer6', seed=1)
# done = False
# obs = env.reset()
# while not done:
#   action = np.random.uniform(low=-1,high=1, size=5)
#   obs, reward, done, info = env.step(action)
#   print(reward)
#######################
##OPENAI GYM SWIMMER V2###


# # # # ####train#######
# The noise objects for TD3
# env = dmc2gym.make(domain_name='myswimmer', task_name='swimmer', seed=1)

# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


# model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log="./swimmer3_trpo_c/") sb2
# model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./swimmer3_td3_c/")
# model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./swimmer3_td3_c/") #base sb3

# model.learn(total_timesteps=100000, log_interval=10)
# model.save("swimmer3_td3_correct")


# del env
# del model
# # # ####test###########
def flatten_obs(obs):
  obs_pieces = []
  for v in obs.values():
    flat = np.array([v]) if np.isscalar(v) else v.ravel()
    obs_pieces.append(flat)
  return np.concatenate(obs_pieces, axis=0)

model = TD3.load("gnn_swimmer3_td3.zip") #SB3 orig
# model = TRPO.load("swimmer3_trpo100k") #SB orig change in swimmer.py state((6,3))

# model = TD3.load("gnn_swimmer_td3.zip")

env = dm_control.suite.load(domain_name='myswimmer', task_name='swimmer')
print(env._task.get_optimal_action(env._physics))
# env = dm_control.suite.load(domain_name='swimmer', task_name='swimmer6') #, seed=1)
time_step = env.reset()

def model_policy(time_step):
  global model
  obs = flatten_obs(time_step.observation)
  action, _ = model.predict(obs)
  # env.step(action)
  print(np.rad2deg(env.task.get_optimal_action(env.physics)))
  print(np.rad2deg(action[0]))
  return action

viewer.launch(env, policy=model_policy)

# #
# Define a uniform random policy.
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# # actions =[]
# width = 300
# height = 300
# video = np.zeros((1, height, 2 * width, 3), dtype=np.uint8)
# max_frame =1000
# j=0
# done = False
# print(env.observation_spec().values())
# print(env.action_spec())
# # # plt.show()
# # while not time_step.last():
# #   obs = flatten_obs(time_step.observation)
# #   action, _ = model.predict(obs)
# #   time_step = env.step(action)
# #   reward=time_step.reward
# #   video[0] = np.hstack([env.physics.render(300, 300, camera_id=0),
# #                         env.physics.render(300, 300, camera_id=1)])
# #     # print(time_step.reward, time_step.discount, time_step.observation)
# # # for i in range(max_frame):
# #   img = plt.imshow(video[0])
# #   plt.pause(0.01)  # Need min display time > 0.0.
# #   plt.draw()
# #   j+=1
# #
# # actions=np.array(actions)
# # print(actions.shape)
# # i=0
# ####################
