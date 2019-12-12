from Env import AirSimEnv
from ACagent import ACAgent
import msgpackrpc
import numpy as np
import os
from utils import plotLearning2

if __name__ == '__main__':
    OUTPUT_PATH = "D:\BandwidthDataset\cooked\\"
    env = AirSimEnv(bandwidth_file_path=os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'))
    num_epochs = 2000
    explore_start = True
    max_epoch_time = 30.
    random_inital_steps = 50
    wait_delta_sec = 0.01
    verbose = True
    train = True
    score_history = []

    agent = ACAgent(alpha=0.00001, beta=0.00005)
    last_action = 1

    for step in range(num_epochs):
        try:
            done = False
            score = 0
            env.reset(explore_start=explore_start, max_epoch_time=max_epoch_time)

            # 获取一个随机初始的状态
            observation, reward, done, info = env.step(last_action=last_action, action=3, max_chunk_time=5)
            print(done)
            while not done:
                action = agent.decide(observation)
                print('action = {} '.format(action))
                next_observation, reward, done, info = env.step(last_action=last_action, action=action, max_chunk_time=5)
                score += reward
                print('score: %.2f' % score)
                # 如果回合刚开始就结束了，就不是靠谱的回合
                if step == 0 and done:
                    if verbose:
                        print('不成功的回合，放弃保存')
                    break

                if train:
                    agent.learn(observation, action, reward, next_observation, done)
                observation = next_observation
                last_action = action
                # 回合结束
                if done:
                    if verbose:
                        print('回合{} 从 {} 到 {} 结束. {}'.format(step, env.start_time, env.end_time, info))
                    break

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            print('episode: ', step, 'score: %.2f' % score,
                  'avg score %.2f' % avg_score)
        # 极少数情况下 AirSim 会停止工作，需要重新启动并连接
        except msgpackrpc.error.TimeoutError:
            print('与 AirSim 连接中断。开始重新链接')
            env.connect()

    filename = 'airsim_ac.png'
    plotLearning2(score_history, filename=filename,window=100)