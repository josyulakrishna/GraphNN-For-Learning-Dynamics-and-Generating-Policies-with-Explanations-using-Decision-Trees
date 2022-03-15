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
env = suite.load(domain_name="swimmer", task_name="swimmer6")
# env = gym.make("Swimmer-v2")
# Iterate over a task set:
#for domain_name, task_name in suite.BENCHMARKING:
# env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.

max_frame = 200
max_episodes = int(sys.argv[1])

width = 480
height = 480
# print("action space", str(env.action_space))
# print("obs space ", str(env.observation_space))
# action_spec = env.action_spec()
# obs_spec = env.observation_spec()
# print("action spec", action_spec)
# print("observation spec", obs_spec)


time_step = env.reset()

actions = np.zeros((201,5))
x = np.arange(0,201,20)

dataset = np.zeros((max_episodes, max_frame+1, 5 + 5 + 18 + 2))
len_dataset = dataset.shape[0]

for idx in tqdm(range(len_dataset)):
    time_step = env.reset()
    video = np.zeros((max_frame, height, width, 3), dtype=np.uint8)
    i = 0
    for j in range(5):
        y = np.random.uniform(-1,1,x.shape)
        tck = interpolate.splrep(x, y, s=0)
        xnew = np.arange(0,201)
        ynew = interpolate.splev(xnew, tck, der=0)

        actions[:,j] = ynew
        # print(len(ynew))
    actions = np.clip(actions, -1, 1)
    record = False
    # print(actions.shape)
    while (time_step.last()-1 )>0:
        action = actions[i]
        time_step = env.step(action)
        #from IPython import embed; embed()
        obs = time_step.observation
        dataset[idx, i,:5] = action
        dataset[idx, i, 5:10] = obs['joints']
        # print(obs['body_velocities'].shape, obs['joints'].shape, obs['to_target'].shape)
        dataset[idx,i,10:28] = obs['body_velocities']
        dataset[idx,i,28:] = obs['to_target']

        if record:
            if i < max_frame:
                video[i] = env.physics.render(height, width, camera_id=0)
        i += 1

    if record:
        writer = imageio.get_writer('test_{}.gif'.format(idx), fps=60)
        for j in range(max_frame):
            writer.append_data(video[j])
        writer.close()

np.save(sys.argv[2], dataset)


plt.plot(actions)
plt.show()
plt.savefig('actions.png')
