from Env import AirSimEnv
import matplotlib.pyplot as plt

env = AirSimEnv()
env.reset(env.reset(explore_start=True, max_epoch_time=True))

env.step(last_action=1, action=2, max_chunk_time=5)

img_sob = env.get_image()
print('plotting')
#plt.imshow(img_sob.astype('uint8'))
#plt.show()\
img_sob.show()