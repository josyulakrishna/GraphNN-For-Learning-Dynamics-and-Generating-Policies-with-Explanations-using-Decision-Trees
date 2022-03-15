import pdb

from dm_control import suite
import myswimmer as swimmer
import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import interpolate
from tqdm import tqdm
import sys
import gym

# Load one task:
# env = swimmer.swimmer6(time_limit = 4) #
env = None
dataset=None

max_frame = 200
max_episodes = int(sys.argv[1])
n_links = int(sys.argv[2])-1

width = 480
height = 480

total_len=0
if int(sys.argv[2])==6:
    env = suite.load(domain_name="myswimmer", task_name="swimmer6")
    obs_spec = env.observation_spec()
    dataset = np.zeros((max_episodes, max_frame + 1, 5 + 5 + 18 + 18))
if int(sys.argv[2])==3:
    env = suite.load(domain_name="myswimmer", task_name="swimmer")
    obs_spec = env.observation_spec()
    for key in obs_spec.keys():
        total_len = obs_spec[key].shape[0]+total_len
    dataset = np.zeros((max_episodes, max_frame + 1, n_links+total_len))



# env = gym.make("Swimmer-v2")
# Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
# env = suite.load(domain_name, task_name)
# action_spec = env.action_spec()
# obs_spec = env.observation_spec()

time_step = env.reset()

actions = np.zeros((201,n_links)) #actions size
x = np.arange(0,201,20)

# dataset = np.zeros((max_episodes, max_frame+1, 5 + 5 + 18 + 18))
len_dataset = dataset.shape[0]
print(len_dataset)

for idx in tqdm(range(len_dataset)):
    time_step = env.reset()
    video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)
    i = 0
    for j in range(n_links):
        y = np.random.uniform(-1,1,x.shape)
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.arange(0,201)
        ynew = interpolate.splev(xnew, tck, der=0)
        actions[:,j] = ynew

    actions = np.clip(actions, -1, 1)
    record = False

    while i<201:
        action = actions[i]
        time_step = env.step(action)
        #from IPython import embed; embed()
        obs = time_step.observation
        dataset[idx,i,:n_links] = action
        index_j=n_links+obs_spec["joints"].shape[0] #4
        index_k = n_links + obs_spec["joints"].shape[0] + obs_spec["abs"].shape[0] #22
        # pdb.set_trace()
        dataset[idx, i, n_links:index_j ]= obs['joints']
        dataset[idx,i,index_j:index_k] = obs['abs']
        dataset[idx,i,index_k:] = obs['body_velocities']
        if record:
            if i < max_frame:
                video[i] = env.physics.render(height, width, camera_id=0)
        i += 1

    if record:
        writer = imageio.get_writer('test_{}.gif'.format(idx), fps=60)
        for j in range(max_frame):
            writer.append_data(video[j])
        writer.close()

np.save(sys.argv[3], dataset)

'''
plt.plot(actions)
plt.savefig('actions.png')
'''
