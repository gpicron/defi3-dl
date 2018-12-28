import numpy as np
from keras.optimizers import RMSprop, Adam
from keras.layers import K

class NetworkPart:
    """ Agent Generic Class
    """

    def __init__(self, inp_dim, out_dim, lr):
        self.inp_dim = inp_dim
        self.out_dim = out_dim

    def finalize_build(self):
        self.predict_with_dropout = K.function([self.model.layers[0].input, K.learning_phase()],
                                                     [self.model.layers[-1].output])


    def predict(self, inp):
        """ Value Prediction
        """
        return self.model.predict(self.reshape(inp))

    def predict_with_uncertainity(self, inp, n_iter=10):
        result = np.zeros((n_iter,) + (self.model.output.shape[1],) )

        for i in range(n_iter):
            p = self.predict_with_dropout((inp, 1))[0]
            result[i,:] = p

        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty


    def reshape(self, x):
        if len(x.shape) == len(self.inp_dim): return np.expand_dims(x, axis=0)
        else: return x