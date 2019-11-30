from Env import AirSimEnv
from DQNagent import DQNAgent
import msgpackrpc
import os
import numpy as np
from utils import plotLearning

if __name__ == '__main__':
    OUTPUT_PATH = "D:\BandwidthDataset\cooked\\"
    env = AirSimEnv(bandwidth_file_path=os.path.join(OUTPUT_PATH, 'bandwidth_modified.txt'))
    num_epochs = 300
    explore_start = True
    max_epoch_time = 30.
    random_inital_steps = 50
    wait_delta_sec = 0.01
    verbose = True

    weight_path = None
    train_conv = True
    train = True
    agent = DQNAgent(weight_path=weight_path, train_conv=train_conv,
        random_inital_steps=random_inital_steps)
    last_action = 1
    ddqn_scores = []
    eps_history = []
    print('start training')
    for step in range(num_epochs):
        try:
            done = False
            score = 0

            # 判断是否用随机动作填充经验库
            random = agent.replayer.count < random_inital_steps
            # 启动新回合，在地图上选择一个地方，并让汽车前进一段
            env.reset(explore_start=explore_start, max_epoch_time=max_epoch_time)

            # 获取一个随机初始的状态
            observation, reward, done, info = env.step(last_action=last_action, action=3, max_chunk_time=5)

            while not done:
                action = agent.decide(observation, random=random)

                print('action = {} '.format(action))
                next_observation, reward, done, info = env.step(last_action=last_action, action=action, max_chunk_time=5)
                score += reward

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
            eps_history.append(agent.epsilon)
            ddqn_scores.append(score)
            avg_score = np.mean(ddqn_scores[max(0, step - 100):(step + 1)])
            print('episode: ', step, 'score: %.2f' % score,' average score %.2f' % avg_score)
            if step % 10 == 0 and step > 0:
                agent.save_models()

        # 极少数情况下 AirSim 会停止工作，需要重新启动并连接
        except msgpackrpc.error.TimeoutError:
            print('与 AirSim 连接中断。开始重新链接')
            env.connect()

    filename = 'airsim-ddqn.png'
    x = [i + 1 for i in range(num_epochs)]
    plotLearning(x, ddqn_scores, eps_history, filename)