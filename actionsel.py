import random

import numpy as np

import util


class ActionSelector:
    def getAction(self, agent, state):
        """
      Returns an action
    """
        util.raiseNotDefined()

    def update(self, agent, state, action, delta):
        pass


class GreedyActionSelector(ActionSelector):

    def getAction(self, agent, state):
        action = agent.getPolicy(state)

        if action is None:
            legal_actions = agent.getLegalActions(state)
            action = random.choice(legal_actions)

        return action


class EGreedyActionSelector(GreedyActionSelector):

    def getAction(self, agent, state):
        explore = util.flipCoin(agent.epsilon)
        if explore:
            legalActions = agent.getLegalActions(state)
            legalActions.remove('Stop')
            return random.choice(legalActions)
        else:
            return GreedyActionSelector.getAction(self, agent, state)


class EntropyActionSelector(GreedyActionSelector):

    def __init__(self):
        ## self.VarQValues[state][action] => var(Q(s,a))
        self.varQValues = dict()

    def getVarQValue(self, state, action):
        """
      Returns variance of Q(state,action)
      Should return very large if we never seen
      a state or (state,action) tuple to express the fact the average is unknown
    """
        return self.varQValues.setdefault(state, dict()).setdefault(action, 1e100)

    def getAction(self, agent, state):
        """
    The idea is when exploring to choose the action that will increase the most the entropy. For that we keep the
    moving average of the delta during the update of Q(s,a).
    """
        explore = util.flipCoin(agent.epsilon)
        if explore:
            legalActions = agent.getLegalActions(state)
            min = 0
            choosen = None

            for action in legalActions:
                VQsa = self.getVarQValue(state, action)
                predictedQs = [
                    agent.getQValue(state, a) if a != action else agent.getQValue(state, a) + agent.alpha * VQsa for a in
                    legalActions]

                predictedQs = np.array(predictedQs) + 1
                probs = predictedQs / np.sum(predictedQs)
                entropy = - np.sum(probs * np.log(probs)) / np.log(len(legalActions))

                if entropy > min:
                    min = entropy
                    choosen = action

            if choosen is None:
                return GreedyActionSelector.getAction(self, agent, state)

            return choosen
        else:
            return GreedyActionSelector.getAction(self, agent, state)

    def update(self, agent, state, action, delta):
        ## Variance of Q(s, a)
        VQsa = self.getVarQValue(state, action)
        VQsa = VQsa + agent.alpha * delta
        VQs = self.varQValues.setdefault(state, dict())
        VQs[action] = VQsa


class UncertainityActionSelector(GreedyActionSelector):

    def getAction(self, agent, state):
        explore = util.flipCoin(agent.epsilon)
        if explore:
            preds = agent.predict_with_uncertainity(state)
            legalActions = agent.getLegalActions(state)

            highestUncertainity = - 1e100
            choosen = None

            for action, (p, u) in preds.items():
                if action in legalActions:
                    if u > highestUncertainity:
                        choosen = action
                        highestUncertainity = u

            if choosen is None:
                return GreedyActionSelector.getAction(self, agent, state)

            return choosen
        else:
            return GreedyActionSelector.getAction(self, agent, state)
