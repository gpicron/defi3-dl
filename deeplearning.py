
from qlearningAgents import ReinforcementAgent
from actionsel import *

class DynamicImporter:
    """"""

    #----------------------------------------------------------------------
    def __init__(self, module_name, class_name):
        """Constructor"""
        module = __import__(module_name)
        self.my_class = getattr(module, class_name)


    def clazz(self):
        return self.my_class

if __name__ == "__main__":
    DynamicImporter("decimal", "Context")

class DeepLearningAgent(ReinforcementAgent):

    def __init__(self, epsilon_max=1, epsilon_min=0.01, epsilon_decay=0.999, gamma=0.8, alpha=0.2, numTraining=0, explorationFunction="EGreedyActionSelector", **args):
        """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.index = 0  # This is always Pacman
        args['epsilon'] = float(epsilon_max)
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining


        ReinforcementAgent.__init__(self, **args)


        self.explorationFunction = util.lookup(explorationFunction, globals())()
        self.final_state = False

    def registerInitialState(self, state):
        self.final_state = False
        ReinforcementAgent.registerInitialState(self, state)

    def final(self, state):
        self.final_state = True
        ReinforcementAgent.final(self, state)
        self.epsilon = max(float(self.epsilon) * self.epsilon_decay, self.epsilon_min)



    def getPolicy(self, state):
        """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
    """
        util.raiseNotDefined()

    def getAction(self, state):
        """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
        action = self.explorationFunction.getAction(self, state)
        self.doAction(state, action)
        return action

