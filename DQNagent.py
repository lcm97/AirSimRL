import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward',
                                            'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in \
                self.memory.columns)

class DQNAgent:
    def __init__(self, gamma=0.99, batch_size=32,
                 replayer_capacity=2000, random_inital_steps=50,
                 weight_path=None, train_conv=True,
                 epsilon=1., min_epsilon=0.1, epsilon_decrease_rate=0.003,
                 q_eval_fname='.\model\\q_eval.h5',q_target_fname='.\model\\q_next.h5'):
        self.action_n = 5
        self.gamma = gamma

        # 经验回放
        self.replayer = DQNReplayer(capacity=replayer_capacity)
        self.batch_size = batch_size
        self.random_inital_steps = random_inital_steps

        # 探索参数
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decrease_rate = epsilon_decrease_rate

        # 搭建网络
        self.evaluate_net = self.build_network(weight_path=weight_path,
                                               train_conv=train_conv)
        self.target_net = self.build_network()
        self.target_net.set_weights(self.evaluate_net.get_weights())

        self.q_target_model_file = q_target_fname
        self.q_eval_model_file = q_eval_fname

    def build_network(self, activation='relu', weight_path=None,
                      train_conv=True, verbose=True):

        input_1 = keras.layers.Input(shape=[1, 1], name='collision')
        input_2 = keras.layers.Input(shape=[1, 1], name='complexity')
        input_3 = keras.layers.Input(shape=[1, 1], name='last_action')
        input_4 = keras.layers.Input(shape=(5, 1), name='throughput')

        input_4_ = keras.layers.Conv1D(125, kernel_size=2, strides=1, activation='relu', padding='same')(input_4)
        input_4_ = keras.layers.MaxPool1D(pool_size=5, strides=1)(input_4_)

        input_ = keras.layers.concatenate([input_1, input_2, input_3, input_4_])

        y = keras.layers.Flatten()(input_)

        # 全连接层
        x = keras.layers.Dropout(0.2)(y)
        z = keras.layers.Dense(128, activation=tf.nn.relu,
                               kernel_initializer= keras.initializers.RandomNormal(stddev=0.01))(x)
        y = keras.layers.Dropout(0.2)(z)
        outputs = keras.layers.Dense(self.action_n,
                                     kernel_initializer= keras.initializers.RandomNormal(stddev=0.01))(y)

        net = keras.Model(inputs=[input_1,input_2,input_3,input_4], outputs=outputs)
        net.compile(optimizer='adam', loss='mse')

        if verbose:
            net.summary()
            #SVG(model_to_dot(net).create(prog='dot', format='svg'))

        if weight_path:
            net.load_weights(weight_path)
            if verbose:
                print('载入网络权重 {}'.format(weight_path))

        return net

    def decide(self, observation, random=False):
        if random or np.random.rand() < self.epsilon:
            return np.random.randint(self.action_n)
        collision = np.expand_dims(observation[0], axis=2)
        complexity = np.expand_dims(observation[1], axis=2)
        last_action = np.expand_dims(observation[2], axis=2)
        throughput = np.expand_dims(observation[3], axis=0)
        qs = self.evaluate_net.predict([collision,complexity,last_action,throughput])
        return np.argmax(qs)


    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                             done)  # 存储经验

        if self.replayer.count < self.random_inital_steps:
            return  # 还没到存足够多的经验，先不训练神经网络

        observations, actions, rewards, next_observations, dones = \
            self.replayer.sample(self.batch_size)  # 经验回放

        next_qs = self.target_net.predict(next_observations)
        next_max_qs = next_qs.max(axis=-1)
        us = rewards + self.gamma * next_max_qs * (1. - dones)
        targets = self.evaluate_net.predict(observations)
        targets[np.arange(us.shape[0]), actions] = us
        self.evaluate_net.fit(observations, targets, verbose=0)

        if done:
            self.target_net.set_weights(self.evaluate_net.get_weights())

        # 减小 epsilon 的值
        self.epsilon -= self.epsilon_decrease_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

    def save_models(self):
        self.evaluate_net.save(self.q_eval_model_file)
        self.target_net.save(self.q_target_model_file)
        print('... saving models ...')