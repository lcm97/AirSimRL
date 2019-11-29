from Env import AirSimEnv
import os
import matplotlib.pyplot as plt
OUTPUT_PATH = "D:\BandwidthDataset\cooked\\"

env = AirSimEnv(bandwidth_file_path= os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'))

env.reset(explore_start=True, max_epoch_time=True)

observation, reward, done, info = env.step(last_action=1, action=2, max_chunk_time=5)
print(observation)
print(reward)

img_sob = env.get_image()
print('plotting')
#plt.imshow(img_sob.astype('uint8'))
#plt.show()\
img_sob.show()