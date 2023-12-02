# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        import copy
        # Initialize mdp
        for state in self.mdp.getStates():
            self.values[state] = 0

        # Value iteration
        for _ in range(self.iterations):
            next_values = copy.deepcopy(self.values)
            for state in self.mdp.getStates():
                candidate_v = []

                if self.mdp.isTerminal(state):
                    next_values[state] = 0
                    continue

                for a in self.mdp.getPossibleActions(state):
                    value = 0
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, a):
                        value += prob * (self.mdp.getReward(state, a, next_state) + self.discount * self.getValue(next_state))
                    candidate_v.append(value)
                
                next_values[state] = max(candidate_v)
            self.values = next_values
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            value += prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.getValue(next_state))
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        from util import PriorityQueue

        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        q_s = PriorityQueue()
        for action in actions:
            q_s.push(action, -self.computeQValueFromValues(state, action))
        return q_s.pop()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        import copy
        # Initialize mdp
        for state in self.mdp.getStates():
            self.values[state] = 0

        counter = 0
        # Value iteration
        while True:
            for state in self.mdp.getStates():
                if counter >= self.iterations:
                    return
                candidate_v = []

                if self.mdp.isTerminal(state):
                    self.values[state] = 0
                    counter += 1
                    continue

                for a in self.mdp.getPossibleActions(state):
                    value = 0
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, a):
                        value += prob * (self.mdp.getReward(state, a, next_state) + self.discount * self.getValue(next_state))
                    candidate_v.append(value)
                
                counter += 1
                
                self.values[state] = max(candidate_v)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {state: set() for state in self.mdp.getStates()}
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[next_state].add(state)

        # Priority queue for prioritized sweeping
        priority_queue = util.PriorityQueue()

        # Initialize priority queue with state priorities
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                diff = abs(self.values[state] - self.computeQValueFromValues(state, self.getAction(state)))
                priority_queue.update(state, -diff)

        # Perform prioritized sweeping iterations
        for _ in range(self.iterations):
            if priority_queue.isEmpty():
                break

            # Get state with highest priority
            state = priority_queue.pop()

            # Update value of the state
            self.values[state] = self.computeQValueFromValues(state, self.getAction(state))

            # Update priorities of predecessors
            for predecessor in predecessors[state]:
                diff = abs(self.values[predecessor] - self.computeQValueFromValues(predecessor, self.getAction(predecessor)))
                if diff > self.theta:
                    priority_queue.update(predecessor, -diff)