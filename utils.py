import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
from math import exp

def plotLearning(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def plotLearning2(scores, filename, x=None, window=5):
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

def plotPower():
    resolution = [32, 64, 128, 256, 320]
    dnn_power = [2628.95, 2677.96, 3066.24,4576.58,4788.67]
    gest_power = [2621,2650.21,2687,2914,3123]
    plt.ylabel('Power consumption(mW)')
    plt.xlabel('Resolution')

    plt.xticks(range(len(resolution)), resolution)
    plt.plot(dnn_power,'o-',label='local')
    plt.plot(gest_power, 'x-',label='offload')
    plt.legend(loc='upper left')

    plt.show()

def debugReward():
    energy = random.uniform(2.,5.)
    complexity = random.uniform(20.,80.)
    resolution = random.choice([32.,64.,128.,256.,320.])
    entropy = random.uniform(0.,0.5)
    print(energy, ' ', complexity, ' ', resolution, ' ', entropy)
    scaled_complexity = 4.8 * complexity - 64
    print(abs(scaled_complexity-resolution))
    k1 = 6.9
    k2 = 1.47
    k3 = 70.0
    # print(k2*exp(k3*((4.8*complexity[0][0]-64)-resolution)))
    reward = - k1 * energy - k2 * abs(scaled_complexity - resolution)- k3 * entropy
    print(reward)
    return reward

def load_img(path, grayscale=False):
    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
    # Returns
        Image as numpy array.
    """
    img = cv2.imread(path,-1)
    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((img.shape[0], img.shape[1], 1))

    return np.asarray(img, dtype=np.float32)
# debugReward()
# i=0
# reward_list = []
# while i<50:
#     reward_list.append(debugReward())
#     i+=1
# x = range(50)
# plt.plot(x,reward_list)
# plt.show()



