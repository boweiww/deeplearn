# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNdata:
    def __init__(self, train_file, test_file, select, action_size):

        df = pd.read_excel(train_file)

        self.b = df.ix[:, select]
        self.df = df.drop(select, 1)
        self.row_num = self.b.shape[0]

        test_df = pd.read_excel(test_file)
        self.state_size = self.df.shape[1]

        self.test_b = test_df.ix[:, select]
        self.test_df = test_df.drop(select, 1)
        self.test_row_num = self.test_b.shape[0]


        # self.state_size = state_size
        self.action_size = action_size -1
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state ):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        state = np.array(state)
        act_values = [0] * (self.action_size + 2)
        # print(act_values)
        if np.random.rand() <= self.epsilon:
            a = random.randint(0,self.action_size+1)
            # print a
            act_values[a] = 1
            return act_values
        act_val = self.model.predict(state)



        # print act_val
        a = np.where(act_val[0] == np.max(act_val[0]))[0][0]
        act_values[a] = 1
        return act_values
        # print act_values
        # return np.argmax(act_values[0])  # returns action


        # print("ininininin")
        return act_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            target = reward
            # print next_state

            # self.model.predict(next_state)
            target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))



            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reward(self, action, expected):
        expected = expected -1
        # print action
        if action[expected] == 0:
            return 0
        action[expected] = 0
        for i in range (self.action_size):

            if np.all((action[i]) == 0):
                continue
            else:
                return 0
        return 1




    def train(self):
        batch_size = 100

        for i in range(self.row_num):
            state = self.df.ix[i].tolist()
            # state = np.reshape(state, [1, state_size])
            # for time in range(500):
                # env.render()
            state = np.array(state)

            state = np.reshape(state, [1, self.state_size])

            action = self.act(state)
            # next_state, reward, done, _ = env.step(action)
            # print action
            reward = self.reward(action, self.b.ix[i].tolist())
            for j in range(self.action_size):
                if action[j] != 0:
                    print ("predict: %d, real value: %d" % (j, self.b.ix[i].tolist()))
                    break



            # next_state = np.reshape(next_state, [1, state_size])
            next_state = self.df.ix[i+1].tolist()
            # print next_state

            next_state = np.array(next_state)

            next_state = np.reshape(next_state, [1, self.state_size])

            self.remember(state, action, reward, next_state)
            # if done:
            #     print("episode: {}/{}, score: {}, e: {:.2}"
            #           .format(e, EPISODES, time, agent.epsilon))
            #     break
            if len(self.memory) > batch_size :
                agent.replay(batch_size)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



if __name__ == "__main__":
    train_file = '/home/bowei/PycharmProjects/test/venv/lib/data-MLP/classification/abalone_train_classification.xlsx'
    test_file = '/home/bowei/PycharmProjects/test/venv/lib/data-MLP/classification/abalone_test_classification.xlsx'

    user_select = 'rings'

    action_size = 13
    # network_wide = [None] * (layers + 1)
    # network_wide[0] = 10
    # network_wide[1] = 5
    # network_wide[2] = 1

    # batch_size = 100

    agent = DQNdata(train_file, test_file, user_select, action_size)
    agent.train()


        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")