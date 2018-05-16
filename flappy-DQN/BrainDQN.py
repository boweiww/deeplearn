# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# original comes from https://github.com/floodsung/DRL-FlappyBird
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
from collections import deque 

class BrainDQN:

    # Hyper Parameters:
    ACTION = 2
    FRAME_PER_ACTION = 1
    GAMMA = 0.99 # decay rate of past observations
    OBSERVE = 100000. # timesteps to observe before training
    EXPLORE = 150000. # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0 # final value of epsilon
    INITIAL_EPSILON = 0.0 # starting value of epsilon
    REPLAY_MEMORY = 50000 # number of previous transitions to remember
    BATCH_SIZE = 32 # size of minibatch

    def __init__(self):
        # init replay memory
        self.replayMemory = deque()
        # init Q network
        self.createQNetwork()
        # init some parameters
        self.timeStep = 0
        self.epsilon = self.INITIAL_EPSILON

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([1600,512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512,self.ACTION])
        b_fc2 = self.bias_variable([self.ACTION])

        # input layer

        # The picture is 80x80 pixels, with 4 colors avaliable
        # We acturally transfer the picture into grayscale
        self.stateInput = tf.placeholder("float",[None,80,80,4])

        # hidden layers
        # First convolution layer and first pooling layer
        h_conv1 = tf.nn.relu(self.conv2d(self.stateInput,W_conv1,4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        # 2nd convolution layer
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

        # 3rd convolution layer
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

        # Reshape can transfer a matrix into an one dimension matrix, by just put [-1] as the second attribute
        # -1 can be used when don't know what value should be putted here
        # I guess the 1600 can be changed to any value that is larger than 256
        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])

        # Twice full connection forward trans.
        # matmul is matrix multiple
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
        self.QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        # declare two placeholder to save input
        self.actionInput = tf.placeholder("float",[None,self.ACTION])
        self.yInput = tf.placeholder("float", [None])

        # not quite clear about the code here, it seemed that the reduce_sum of the
        # QValue and actionInput is calculated
        Q_action = tf.reduce_sum(tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_action))
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # saving and loading networks
        saver = tf.train.Saver()
        self.session = tf.InteractiveSession()

        # run the tensorflow
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        # check if there is value avaliable
        if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(self.session, checkpoint.model_checkpoint_path)
                print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
                print "Could not find old network weights"

    def trainQNetwork(self):

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,self.BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        # feed_dict is fill the sequence like a directory
        y_batch = []
        # eval is another way to run the calculate
        QValue_batch = self.QValue.eval(feed_dict={self.stateInput:nextState_batch})

        # Here is the place that Q learning is done
        # if the bird is died, save the reward, or increase the reward
        for i in range(0,self.BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })

        # save network every 10000 iteration
        if self.timeStep % 10000 == 0:
            saver.save(self.session, 'saved_networks/' + 'network' + '-dqn', global_step = self.timeStep)


    def setPerception(self,nextObservation,action,reward,terminal):
        newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)

        # structure is same as the previous difined one.
        # cur state, action, reward, newstate, terminal.
        # add the current information into the memory.
        self.replayMemory.append((self.currentState,action,reward,newState,terminal))

        # if the replay take too much memory, move it left(which means remove the oldest one)
        if len(self.replayMemory) > self.REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.timeStep > self.OBSERVE:
            # Train the network
            # This program will first observe enough amount of examples so that the batch can be
            # enough effective, then begin training
            self.trainQNetwork()

        # point to the next state
        self.currentState = newState
        self.timeStep += 1

    def getAction(self):
        QValue = self.QValue.eval(feed_dict = {self.stateInput:[self.currentState]})[0]
        action = np.zeros(self.ACTION)
        action_index = 0
        # action part of the Q learning
        # get the random number and compare with the epsilon
        # to decide whether we should take the random action or work the greedy algorithm
        # I guess the FRAME_PER_ACTION is the minimum element of an action
        if self.timeStep % self.FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.ACTION)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1 # do nothing

        # change episilon
        # under the current case, the program will only do the greedy one
        # no random(or to say, explore under RL) will be done
        # we can change that by changing the value of the very first
        # This model should be already good enough so that not need to do any explore
        if self.epsilon > self.FINAL_EPSILON and self.timeStep > self.OBSERVE:
            self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON)/self.EXPLORE

        return action

    # Called in the main function,
    def setInitState(self,observation):
        self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

    # get weight variable by the shape
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    # get bias variable by the shape
    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    # do the convolution
    # with size = 1 and pace = stride
    # therefore under the first convolution layer the size of the matrix will be transferred from 80x80
    # to 20x20, many value is missed
    # (if take the larger size of the convolution will cause a better result?)
    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    # get the max pooling layer
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")