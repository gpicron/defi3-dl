import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Concatenate, concatenate, Dropout, Dot
from keras.optimizers import Adam
from .networkpart import NetworkPart

class Critic(NetworkPart):
    """ Critic for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, shared_network, lr):
        NetworkPart.__init__(self, inp_dim, out_dim, lr)
        self.action = Input(shape=(self.out_dim,), name="action")
        self.model = self.addHead(shared_network)

    def addHead(self, shared):
        """ Assemble Critic network to predict value of each state
        """
        x = shared.output
        x = Dense(self.out_dim, activation='linear', name="critics")(x)
        out = Dot(axes=-1)([x, self.action])
        return Model([shared.input[0], shared.input[1], shared.input[2], self.action], out)

    def reshape(self, x):
        states, mems, poses, actions = x
        if len(states.shape) == len(self.inp_dim): states = np.expand_dims(states, axis=0)
        if len(poses.shape) == 1: poses = np.expand_dims(poses, axis=0)
        if len(mems.shape) == 1: mems = np.expand_dims(mems, axis=0)
        if len(actions.shape) == 1: actions = np.expand_dims(actions, axis=0)

        return [states, poses, mems, actions]