#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dmc2gym
import sys
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from stable_baselines3.common.results_plotter import load_results

def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model at {} timesteps".format(x[-1]))
                    print("Saving new best model to {}.zip".format(self.save_path))
                  self.model.save(self.save_path)

        return True

from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir='td3_models')
        
env = dmc2gym.make(domain_name='myswimmer', task_name='swimmer', seed=1)
from stable_baselines3 import PPO, TD3
model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="./swimmer3_td3_c/") #base sb3
# with ProgressBarManager(10000) as progress_callback:
model.learn(total_timesteps=110000, log_interval=10)
model.save("swimmer3_td3_correct")


# In[ ]:


from stable_baselines3 import PPO, TD3
model = TD3.load("swimmer3_td3_correct")


# In[ ]:


import dmc2gym
import sys
import dm_control.suite
import collections
import numpy as np


env_opt = dm_control.suite.load(domain_name='myswimmer', task_name='swimmer', task_kwargs={'random': 10})


# env = dmc2gym.make(domain_name='myswimmer', task_name='swimmer', seed=1)

def sanitise_action(action):
    if action<-1:
        action = -1
    elif action > 1:
        action = 1
    return action

def flatten_obs(obs):
  obs_pieces = []
  for v in obs.values():
    flat = np.array([v]) if np.isscalar(v) else v.ravel()
    obs_pieces.append(flat)
  return np.concatenate(obs_pieces, axis=0)


init_dataset = []
init_targets = []
#env_opt.reset()
init_actions = []
init_rewards = []

time_step = env_opt.reset()
obs = env_opt.observation_spec()

while not time_step.last():
    obs = flatten_obs(time_step.observation)
    action, _ = model.predict(obs)
    # action_opt = env_opt._task.get_optimal_action(env_opt._physics)
    init_dataset.append(obs)
    init_targets.append(1 if action[0]>0 else 0)
    #action[0] = sanitise_action(action_opt)
    # obs, reward, done, info = env.step(action[0])
    # time_Step = env_opt.step(action[0])
    time_step = env_opt.step(action)
    reward=time_step.reward
    obs = flatten_obs(time_step.observation)
    # print(obs)
    
    init_actions.append(action)
    # opt_actions.append(action_opt)
    init_rewards.append(reward)
    
opt_dataset = []
opt_targets = []
#env_opt.reset()
opt_actions = []
opt_policy_actions = []
opt_rewards = []

env_opt = dm_control.suite.load(domain_name='myswimmer', task_name='swimmer', task_kwargs={'random': 10})

time_step = env_opt.reset()
obs = env_opt.observation_spec()

while not time_step.last():
    obs = flatten_obs(time_step.observation)
    action, _ = model.predict(obs)
    action_opt = env_opt._task.get_optimal_action(env_opt._physics)
    opt_dataset.append(obs)
    if using_optimal is True:
        opt_targets.append(1 if action[0]>0 else 0)
    action[0] = sanitise_action(action_opt)
    time_step = env_opt.step(action)
    reward=time_step.reward
    obs = flatten_obs(time_step.observation)
    # print(obs)
    # dataset.append(obs)
    
    opt_actions.append(action)
    # opt_actions.append(action_opt)
    opt_rewards.append(reward)


# In[ ]:


import csv
with open('activity_optactive_reward-100000', 'w') as csvfile:
    writer=csv.writer(csvfile, delimiter=',')
    writer.writerows(zip(init_dataset,init_actions, init_rewards, init_targets))


# In[ ]:


with open('activity_optactive_reward2-100000', 'w') as csvfile:
    writer=csv.writer(csvfile, delimiter=',')
    writer.writerows(zip(opt_dataset,opt_actions, opt_rewards, opt_targets))


# In[ ]:


from sklearn import tree
clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
clf_init = clf.fit(init_dataset, init_targets)


# In[ ]:


from sklearn.tree import export_text
export_text(clf, feature_names=['1','2','3','4','5','6','7','8','9','10',
                               '11','12','13','14','15','16','17','18','19','20',
                               '21','22','23','24','25','26','27','28','29','30',
                               '31','32','33','34','35','36','37','38'])


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf)
fig.savefig('initialtree-110000.png')
# plt.savefig('initialtree.png',format='eps',bbox_inches = "tight")


# In[ ]:


clf = tree.DecisionTreeClassifier(random_state=0, max_depth=3)
clf_opt = clf.fit(opt_dataset, opt_targets)
export_text(clf, feature_names=['1','2','3','4','5','6','7','8','9','10',
                               '11','12','13','14','15','16','17','18','19','20',
                               '21','22','23','24','25','26','27','28','29','30',
                               '31','32','33','34','35','36','37','38'])
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf)
fig.savefig('optimaltree-110000.png')


# In[ ]:


import bitarray as bt
true_positive = 0
false_positive = 0
opt_classes = clf_opt.predict(opt_dataset)
init_classes = clf_init.predict(opt_dataset)
accuracy = sum(1 for x,y in zip(opt_classes,init_classes) if x == y) / len(init_classes)
from sklearn.metrics import precision_recall_fscore_support as pr
bPrecis, bRecall, bFscore, bSupport = pr(init_classes, opt_classes, average='binary')


# In[ ]:


accuracy


# In[ ]:


file = open("performance-110000.txt", "w")
file.write("Timesteps = 110000 \nAccuracy = " + str(accuracy) + "\n" +"Precision = "+ str(bPrecis) + "\n"+"Recall = "+ str(bRecall) )
file.close()


# In[ ]:





# In[ ]:


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

