from keras.utils import to_categorical

from ExperienceMemory import ExperienceMemory, Experience
from deeplearning import DeepLearningAgent, DynamicImporter
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qlearningAgents import *

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optimizers, Model
import keras
from collections import deque
from keras.utils import plot_model

class PacmanDLAgent(DeepLearningAgent):

    def __init__(self, load_weights=None, save_path=".", lr=0.01, algorithm="A2C.A2C", **args):
        DeepLearningAgent.__init__(self, **args)

        self.lr = float(lr)
        self.save_path = save_path
        self.load_weights = load_weights
        self.output_size = len(PacmanDLAgent.actions)
        self.replayLoss = ExperienceMemory(100000)
        self.replayWon = ExperienceMemory(100000)
        self.replayWonGame = ExperienceMemory(100000)
        self.final_state = False
        self.model = None
        self.width = None
        self.height = None
        self.episode_memory = []
        self.save_period = 1000
        self.scores = []
        args = algorithm.rsplit('.')
        self.algorithm = DynamicImporter(args[0], args[1])

    actions = [
        Directions.NORTH,
        Directions.SOUTH,
        Directions.EAST,
        Directions.WEST
    ]

    action_to_i = dict([(a, i) for (i,a) in enumerate(actions)])

    # begin a new episode
    def registerInitialState(self, state):
        self.width = state.data.layout.width
        self.height = state.data.layout.height

        # initialize the model
        if self.model is None:
            #matrices = self.extract_state_matrices(state)
            matrices = self.extract_partial_state_matrices(state)
            self.input_size = matrices.shape
            self.model = self.algorithm.clazz()(self.output_size, self.input_size, lr=self.lr)

            if self.load_weights is not None:
                self.model.load("%s/a2c-%sx%s-%s" % (self.save_path, self.width, self.height, self.load_weights))
                frame = pd.read_pickle("%s/a2c-%sx%s-%s-results.pkl" % (self.save_path, self.width, self.height, self.load_weights))
                self.scores.extend(frame.to_dict('records'))


        DeepLearningAgent.registerInitialState(self, state)

        self.short_memory = np.zeros((256,))



    def final(self, state):
        DeepLearningAgent.final(self, state)
        totalReward = np.sum(np.array([e.reward for e in self.episode_memory]))

        # compute discounted rewards for episode
        cumul_r = 0

        next = None

        gameWon = None

        for exp in reversed(self.episode_memory):
            if gameWon is None:
                gameWon = exp.gameWon
            cumul_r = exp.reward + cumul_r * self.discount
            exp.reward = cumul_r
            exp.episode = self.episodesSoFar
            exp.weight = (exp.weight - cumul_r)**2
            exp.gameWon = gameWon

            if gameWon:
                self.replayWonGame.append(exp)
            elif exp.reward > 0:
                self.replayWon.append(exp)
            else:
                self.replayLoss.append(exp)
            if next is not None:
                next.previous = exp
            next = exp

        next.previous = None

        id = self.episodesSoFar + (0 if self.load_weights is None else int(self.load_weights))
        row = {'Episode': id, 'Score': state.getScore(), 'Steps': len(self.episode_memory), 'Reward': totalReward }

        batch = []
        weights = []

        memories = [self.replayWon, self.replayLoss, self.replayWonGame]

        class_weights = np.array([len(m) for m in memories])
        class_weights = np.sum(class_weights) / class_weights

        max_len = 1#np.min([len(m) for m in memories])

        if max_len > 0:
            for i, mem in enumerate(memories):
                if len(mem) == 0: continue

                class_weight = class_weights[i]
                mini_batch = mem.sample(min(len(mem), 16*max(2,len(self.episode_memory))))
                ep = mem.last().episode
                mini_weights = [0.999**(ep - e.episode) * class_weights for e in mini_batch]

                batch.extend(mini_batch)
                weights.extend(mini_weights)

            states = [e.state for e in batch]
            short_memories = [e.internal_state for e in batch]
            #short_memories = []
            #for e in batch:
                # serie = []
                # pe = e
                # while pe.previous is not None:
                #     pe = pe.previous
                #     serie.append(pe)
                # knowledge = np.zeros((1,1,32))
                # for pe in serie:
                #     knowledge=self.model.new_internal_state(pe.state, knowledge)
                # short_memories.append(knowledge)

            actions = [e.action for e in batch]
            rewards = [e.reward for e in batch]

            r, errors = self.model.train_models(states, short_memories, actions, rewards, True, weights)

            for i in range(len(batch)):
                batch[i].weight = errors[i]

            if r is not None:
                row.update(r)


        self.scores.append(row)

        self.episode_memory.clear()

        if (self.episodesSoFar % self.save_period) == 0:

            self.model.save("%s/a2c-%sx%s-%s" % (self.save_path, self.width, self.height, id))

            if len(self.scores) > 0:
                frame = pd.DataFrame(self.scores).set_index('Episode')

                frame.to_pickle("%s/a2c-%sx%s-%s-results.pkl" % (self.save_path, self.width, self.height, id))
                frame.rolling(200).mean().plot(subplots=True, figsize=(18, 12))
                #plt.show()
                plt.savefig("%s/a2c-%sx%s-%s-results.png" % (self.save_path, self.width, self.height, id))



    def getPolicy(self, state):
        probs = np.copy(self.action_values)

        legalActions = set(self.getLegalActions(state))
        for i, a in enumerate(PacmanDLAgent.actions):
            if a not in legalActions:
                probs[i] = 0

        probs_sum = probs.sum()
        if probs_sum == 0:
            return self.getLegalActions(state)[0]

        #chosen = PacmanDLAgent.actions[np.random.choice(np.arange(self.output_size), 1, p=probs / probs_sum)[0]]
        chosen = PacmanDLAgent.actions[np.argmax(probs)]

        return chosen

    def getAction(self, state):
        self.state_matrices = (self.extract_partial_state_matrices(state), self.extract_pacman_pos(state))

        self.action_values, self.next_short_memory = self.model.predict_action(self.state_matrices, self.short_memory)

        #action = self.explorationFunction.getAction(self, state)
        action = self.getPolicy(state)
        self.doAction(state, action)

        return action


    def update(self, state, action, nextState, reward):
        #state_matrices = self.extract_state_matrices(state)
        #state_matrices = (self.extract_local_state_matrices(state), self.extract_pacman_pos(state))


        action_one_hot = to_categorical(PacmanDLAgent.action_to_i[action], len(PacmanDLAgent.actions))

        gameWon = None

        if reward > 20:
            reward = 5.    # Eat ghost   (Yum! Yum!)
        elif reward > 0:
            reward = 10.    # Eat food    (Yum!)
        elif reward < -10:
            reward = -500. + len(self.episode_memory)  # Get eaten   (Ouch!) -500
            gameWon =  False
        elif reward < 0:
            reward = 0.    # Punish time (Pff..)
        if self.final_state and reward > 0:
            reward = 500. - len(self.episode_memory)
            gameWon = True

        experience = Experience(self.state_matrices, action_one_hot, nextState, reward)
        experience.internal_state = self.short_memory
        experience.weight = self.action_values[self.action_to_i[action]] #temporarly store value
        experience.gameWon = gameWon

        self.episode_memory.append(experience)

        self.short_memory = self.next_short_memory

    def extract_pacman_pos(self, state):
        width, height = self.width, self.height
        x, y = state.getPacmanPosition()

        return (x/width * 2 - 1, y/height * 2 - 1)

    def extract_local_state_matrices(self, state):
        """ Return wall, ghosts, food, capsules matrices around pacman"""

        def get_from_grid(grid):
            x,y = pos
            rx, ry = view_range
            ox, oy = int(rx / 2),int(ry / 2)
            matrix = np.zeros(view_range, dtype=np.int8)
            for i in range(rx):
                for j in range(ry):
                    # Put cell vertically reversed in matrix
                    at_x, at_y = x - ox + i, y - oy + j
                    if at_x < 0 or at_x >= width or at_y < 0 or at_y >= height:
                        matrix[-1-j][i] = -1
                    elif grid[at_x][at_y]:
                        matrix[-1-j][i] = 1
            return matrix

        def get_from_items(items):
            matrix = np.zeros(view_range, dtype=np.int8)
            x,y = pos
            rx, ry = view_range
            ox, oy = int(rx / 2),int(ry / 2)

            for ax, ay in items:
                rel_x, rel_y = int(ax - x + ox), int(ay - y + oy)
                if rel_x >= 0 and rel_x < rx and rel_y >= 0 and rel_y < ry:
                    matrix[-rel_y-1][rel_x] = 1

            return matrix


        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            return get_from_grid(state.data.layout.walls)

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            items = [agentState.configuration.getPosition() for agentState in state.data.agentStates
                     if not agentState.isPacman and not agentState.scaredTimer > 0]

            return get_from_items(items)

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            items = [agentState.configuration.getPosition() for agentState in state.data.agentStates
                     if not agentState.isPacman and agentState.scaredTimer > 0]

            return get_from_items(items)

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            return get_from_grid(state.data.food)

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """

            return get_from_items(state.data.layout.capsules)

        # Create observation matrix as a combination of
        # wall, ghost, food and capsule matrices
        width, height = self.width, self.height
        pos = state.getPacmanPosition()
        view_range = (5,5)
        observation = np.zeros((6,) + view_range)

        observation[0] = getWallMatrix(state)
        observation[1] = getGhostMatrix(state)
        observation[2] = getScaredGhostMatrix(state)
        observation[3] = getFoodMatrix(state)
        observation[4] = getCapsulesMatrix(state)
        observation[5][2][2] = 1

        observation = np.swapaxes(observation, 0, 2)

        return np.array(observation)

    def extract_partial_state_matrices(self, state):
        """ Return wall, ghosts, food, capsules matrices around pacman"""

        def get_from_grid(grid):
            x,y = pos
            rx, ry = view_range
            ox, oy = int(rx / 2),int(ry / 2)
            matrix = np.zeros((width, height), dtype=np.int8)
            for i in range(rx):
                for j in range(ry):
                    # Put cell vertically reversed in matrix
                    at_x, at_y = x - ox + i, y - oy + j
                    if at_x < 0 or at_x >= width or at_y < 0 or at_y >= height:
                        continue
                    if grid[at_x][at_y]:
                        matrix[at_x][at_y] = 1
            return matrix

        def get_from_items(items):
            matrix = np.zeros((width, height), dtype=np.int8)
            x,y = pos
            rx, ry = view_range
            ox, oy = int(rx / 2),int(ry / 2)

            for ax, ay in items:
                rel_x, rel_y = int(ax - x + ox), int(ay - y + oy)
                if rel_x >= 0 and rel_x < rx and rel_y >= 0 and rel_y < ry:
                    matrix[int(ax)][int(ay)] = 1

            return matrix


        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            return get_from_grid(state.data.layout.walls)

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            items = [agentState.configuration.getPosition() for agentState in state.data.agentStates
                     if not agentState.isPacman and not agentState.scaredTimer > 0]

            return get_from_items(items)

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            items = [agentState.configuration.getPosition() for agentState in state.data.agentStates
                     if not agentState.isPacman and agentState.scaredTimer > 0]

            return get_from_items(items)

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            return get_from_grid(state.data.food)

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """

            return get_from_items(state.data.layout.capsules)

        # Create observation matrix as a combination of
        # wall, ghost, food and capsule matrices
        width, height = self.width, self.height
        pos = state.getPacmanPosition()
        view_range = (5,5)
        observation = np.zeros((6,width,height))

        observation[0] = getWallMatrix(state)
        observation[1] = getGhostMatrix(state)
        observation[2] = getScaredGhostMatrix(state)
        observation[3] = getFoodMatrix(state)
        observation[4] = getCapsulesMatrix(state)
        for x in range(max(0,pos[0]-2), min(width, pos[0]+3)):
            for y in range(max(0,pos[1]-2), min(height, pos[1]+3)):
                observation[5][x][y] = 1

        observation = np.swapaxes(observation, 0, 2)

        return np.array(observation)

    def extract_state_matrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height = self.width, self.height
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return np.array(observation)