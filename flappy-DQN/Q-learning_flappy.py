# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# original comes from https://github.com/floodsung/DRL-FlappyBird
# Date: 2016.3.21
# -------------------------

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN import BrainDQN
import numpy as np

# preprocess raw image to 80*80 gray image
def preprocess(observation):

    # get the backgroud grayscale before the process
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def playFlappyBird():
    # Step 1: init BrainDQN
    brain = BrainDQN()
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0])  # do nothing (I think it acturally takes a random act from 0 or 1)

    # get all the returned values
    observation0, reward0, terminal = flappyBird.frame_step(action0)

    #transfer into grayscale
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)

    # Initialize the state by the first observation
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()

        # pass action in and get the return value
        nextObservation,reward,terminal = flappyBird.frame_step(action)

        # get grayscale
        nextObservation = preprocess(nextObservation)

        # Record this state, put it into experience to be avaliable for the batch
        # and train the model by Q-learning
        brain.setPerception(nextObservation,action,reward,terminal)

def main():
    playFlappyBird()

if __name__ == '__main__':
    main()