from Env import AirSimEnv
from ACagent import ACAgent
from DQNagent import DQNAgent
import os
import matplotlib.pyplot as plt

OUTPUT_PATH = "D:\BandwidthDataset\cooked\\"

env = AirSimEnv(bandwidth_file_path= os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'))
env.reset(explore_start=True, max_epoch_time=True)

observation, reward, done, info = env.step(last_action=1, action=2, max_chunk_time=5)
print(observation)

agent = ACAgent(alpha=0.00001, beta=0.00005)

print(agent.decide(observation))

random_inital_steps = 50
weight_path = None
train_conv = True
train = True

agent = DQNAgent(weight_path=weight_path, train_conv=train_conv,
                 random_inital_steps=random_inital_steps)

print(agent.decide(observation))


# collision = np.expand_dims(observation[0], axis=2)
# complexity = np.expand_dims(observation[1], axis=2)
# last_action = np.expand_dims(observation[2], axis=2)
#
# print(observation[3])
# throughput = np.expand_dims(observation[3], axis=0)
# print(throughput)
#
#
# probabilities = agent.policy.predict([collision,complexity, last_action,throughput])
# print(probabilities)
# print(reward)
#
# img_sob = env.get_image()
# print('plotting')
# #plt.imshow(img_sob.astype('uint8'))
# #plt.show()\
# img_sob.show()