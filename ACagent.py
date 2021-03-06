from keras import backend as K
from keras.layers import Dense, Activation, Input, Conv2D, MaxPooling2D, Flatten, Conv1D, concatenate,MaxPool1D
from keras.models import Model, load_model
from keras.optimizers import Adam
import numpy as np

class ACAgent:
    def __init__(self, alpha, beta, gamma=0.99, action_n = 5,
                 layer1_size=1024, layer2_size=512, input_shape=(59, 255, 3)):
        self.gamma = gamma
        self.alpha = alpha # actor学习率
        self.beta = beta   # critic学习率
        self.input_shape = input_shape
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = action_n

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(action_n)]


    def build_actor_critic_network(self):
        delta = Input(shape=[1])
        input_1 = Input(shape=[1, 1], name='collision')
        input_2 = Input(shape=[1, 1], name='complexity')
        input_3 = Input(shape=[1, 1], name='last_action')
        input_4 = Input(shape=(5, 1), name='throughput')

        input_4_ = Conv1D(5, kernel_size=2, strides=1, activation='relu', padding='same')(input_4)
        input_4_ = MaxPool1D(pool_size=5, strides=1)(input_4_)

        input_ = concatenate([input_1, input_2, input_3, input_4_])
        x = Flatten()(input_)
        flat_input = Activation('relu')(x)

        dense1 = Dense(self.fc1_dims, activation='relu')(flat_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(input=[input_1,input_2,input_3,input_4, delta], output=[probs])

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(input=[input_1,input_2,input_3,input_4], output=[values])

        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[input_1,input_2,input_3,input_4], output=[probs])
        print(policy.summary())
        return actor, critic, policy


    def interpret_observation(self,observation):
        collision = np.expand_dims(observation[0], axis=2)
        complexity = np.expand_dims(observation[1], axis=2)
        last_action = np.expand_dims(observation[2], axis=2)
        throughput = np.expand_dims(observation[3], axis=0)
        return [collision,complexity,last_action,throughput]


    def decide(self, observation):
        probabilities = self.policy.predict(self.interpret_observation(observation))[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action


    def learn(self, state, action, reward, state_, done):


        state = self.interpret_observation(state)
        state_ = self.interpret_observation(state_)

        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        act_state = [state[0],state[1],state[2],state[3],delta]

        self.actor.fit(act_state, actions, verbose=0)

        self.critic.fit(state, target, verbose=0)