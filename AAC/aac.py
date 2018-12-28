import curses
import random

import math
import numpy as np
from keras.constraints import MinMaxNorm, Constraint
from keras.engine import Layer
from keras.initializers import Ones
from keras.optimizers import Adam

from tqdm import tqdm
from keras.models import Model
from keras import regularizers, Sequential
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Conv3D, Conv2D, Reshape, Dropout, K, Concatenate, Add, Multiply, Lambda, \
    Dot, Conv2DTranspose, RepeatVector, BatchNormalization, Activation, ZeroPadding2D

from .critic import Critic

class Between0And1(Constraint):
    """Constrains the weights to be non-negative.
    """

    def __call__(self, w):
        return K.clip(w, 0., 1.)


class UpdateKnowledgeLayer(Layer): # a UpdateKnowledgeLayer layer
    def __init__(self, **kwargs):
        super(UpdateKnowledgeLayer
              , self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        w_dim = input_shape[1][1:]
        self.W = self.add_weight(name="alpha", shape=w_dim, # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True, constraint=Between0And1())
        super(UpdateKnowledgeLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        output = x[0] * self.W + x[1] * (1 - self.W)
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

def apply_affine(x):
    import tensorflow as tf
    return tf.contrib.image.translate(x[0], x[1] * 3.5, interpolation="BILINEAR")

def apply_affine_output_shape(input_shapes):
    return input_shapes[0]


class AAC:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, gamma = 0.99, lr = 0.01):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.r_channels = 32
        self.gamma = gamma

        self.buildNetwork()

        optimizer = Adam(lr=lr)

        self.train_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        print(self.train_model.summary())

        self.prediction_model.compile(optimizer, loss="mae")

    def build_tower(self, frame, pose):

        net = Conv2D(filters=self.r_channels // 2, kernel_size=3, strides=1, padding="same", activation="relu")(frame)
        net = Conv2D(filters=self.r_channels, kernel_size=3, strides=1, padding="same", activation=None)(net)
        #skip1 = Conv2D(filters=self.r_channels // 2,kernel_size=1,strides=1, padding="same")(net)
        #net = Conv2D(filters=self.r_channels // 2,kernel_size=3, strides=1, padding="same", activation="relu")(net)
        #net = Add()([net,skip1])
        #net = Conv2D(filters=self.r_channels, kernel_size=2, strides=2, padding="valid", activation="relu")(net)
        net = BatchNormalization()(net)
        net = Activation("relu")(net)

        # tile the poses to match the embedding shape
        height, width = net.shape[1], net.shape[2]

        net = ZeroPadding2D((1, 1))(net)
        net = Lambda(apply_affine, apply_affine_output_shape)([net, pose])

        #pose = self.broadcast_pose(pose, height, width)

       # net = Concatenate(axis=3)([net,pose])

        #skip2 = Conv2D(filters=self.r_channels // 2, kernel_size=1, strides=1, padding="SAME")(net)
        #net = Conv2D(filters=self.r_channels // 2, kernel_size=3, strides=1, padding="SAME", activation="relu")(net)

        #net = Add()([net,skip2])

        #net = Conv2D(filters=self.r_channels, kernel_size=3, strides=1, padding="SAME", activation="relu")(net)

        #net = Conv2D(filters=self.r_channels, kernel_size=1, strides=1, padding="SAME", activation="relu")(net)
        #net = BatchNormalization()(net)

        net = Flatten()(net)
        net = Dense(256, activation="relu")(net)

        return net


    def broadcast_pose(self, pose, height, width):
        pose = Reshape((1,1,2))(pose)
        pose = Lambda(lambda pose: K.tile(pose, [1, height, width, 1]))(pose)
        return pose


    def buildNetwork(self):
        """ Assemble shared layers
        """
        self.input_env = Input(self.env_dim, name="env_state")
        self.input_pos = Input((2,), name="env_pos")
        self.input_knowledge = Input((256,), name="prev_knowledge")
        self.input_action = Input(shape=(self.act_dim,), name="action")

        knowledge_update = self.build_tower(self.input_env, self.input_pos)

        self.update_knowledge_layer = UpdateKnowledgeLayer()
        self.knowledge = self.update_knowledge_layer([self.input_knowledge, knowledge_update])

        #height, width = int(self.knowledge.shape[1]), int(self.knowledge.shape[2].value)

#        pose_channel = Lambda(lambda x: K.ones((K.shape(x)[0],1,1,1)) )(self.knowledge)
#       pose_channel = ZeroPadding2D((3, 3))(pose_channel)
#        pose_channel = Lambda(apply_affine, apply_affine_output_shape)([pose_channel, self.input_pos])


        #query = Concatenate(axis=-1)([self.knowledge, self.input_pos])
#        query = Conv2D(filters=self.r_channels, kernel_size=3, strides=1, padding="SAME", activation=None)(query)
#        query = BatchNormalization()(query)
#        query = Activation("relu")(query)



        values = Dense(4, activation="linear")(self.knowledge)

        self.prediction_model = Model([self.input_env, self.input_pos, self.input_knowledge], [values, self.knowledge])

        predicted_for_action = Dot(axes=-1)([values, self.input_action])

        self.train_model = Model(inputs=[self.input_env, self.input_pos, self.input_knowledge, self.input_action], outputs=[predicted_for_action])

        pass

    def save(self, filename):
        print("saving weight %s" % filename)
        self.train_model.save_weights("%s.weights" % filename, True)

    def load(self, filename):
        print("loading weights %s" % filename)
        self.train_model.load_weights("%s.weights" % filename)


    def predict_action(self, s, mem):
        inputs = ([np.array([s[0]]), np.array([s[1]]), np.array([mem])])
        values, knowledge = self.prediction_model.predict(inputs)

        values = values - np.min(values)

        return values[0] / np.sum(values), knowledge[0]


    def predict_with_uncertainity(self, s, mem, n_iter=50):
        return self.actor.predict_with_uncertainity([s, mem], n_iter)

    def train_models(self, states, mems, actions, discounted_rewards, done, weights=None):
        mems = np.array(mems)

        env_view = np.array([s[0] for s in states])
        poses = np.array([s[1] for s in states])
        actions = np.array(actions)
        #weights = np.array(weights)
        #where_are_NaNs = np.isnan(weights)
        #weights[where_are_NaNs] = 1e-1000
        discounted_rewards = np.array(discounted_rewards)

        #mems = np.zeros_like(mems)

        inputs = [env_view, poses, mems, actions]
        hc = self.train_model.fit(inputs, discounted_rewards, batch_size=32, verbose=0, epochs=3)

        #print(hc.history['loss'])

        result = dict([('Train_'+k, v[-1]) for (k,v) in hc.history.items()])
        result['update_rate'] = np.mean(self.update_knowledge_layer.get_weights()[0])

        #write, _ = self.get_keep_write([env_view, poses, mems, 0])

        #result['write_mean'] = np.mean(write)
        #result['write_std'] = np.std(write)
        #result['keep_mean'] = np.mean(keep)
        #result['keep_std'] = np.std(keep)
        #dummy = np.zeros_like(discounted_rewards)

        #inputs = self.actor.reshape([states, mems])
        #ha = self.actor_critic.fit(inputs, dummy, batch_size=32, verbose=0, epochs=1)

        #result.update(dict([('Train_actor_'+k, v[-1]) for (k,v) in ha.history.items()]))

        return result