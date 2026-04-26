# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()  # base score from the game state

        # Reward proximity to nearest food
        foodList = newFood.asList()
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 1.0 / minFoodDist

        # Penalize proximity to active ghosts; reward chasing scared ghosts
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                score += 2.0 / (dist + 1)  # chase scared ghosts
            else:
                if dist <= 1:
                    score -= 500  # imminent danger
                else:
                    score -= 2.0 / dist  # mild repulsion

        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        def minimax(state, agentIndex, depth):
            """
            Return the minimax value of `state` for the active `agentIndex`.

            Depth semantics for this project:
            - One search ply = Pacman moves once AND every ghost responds once.
            - `depth` is decremented only when the turn wraps back to Pacman.

            Agent roles:
            - Pacman (agent 0) is the maximizing player.
            - Each ghost (agent >= 1) is a minimizing player.
            """
            # if the game is over or we've reached the maximum search depth, return the evaluation of the state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Get total number of agents (Pacman + ghosts)
            numAgents = state.getNumAgents()
            # Get the list of moves the current agent is allowed to make.
            legalActions = state.getLegalActions(agentIndex)

            # Cycle agents: after the last ghost moves, it's Pacman's turn again
            nextAgent = (agentIndex + 1) % numAgents
            # Only decrease depth when it's Pacman's turn again
            #     one full ply is complete after all agents have moved once
            # "ply" being a single turn for all agents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # The list of resulting states after the current agent takes each legal action
            successors = [state.generateSuccessor(agentIndex, a) for a in legalActions]

            # Minimax Decision Tree, Pacman maximizes, ghosts minimize
            if (
                agentIndex == 0
            ):  # Pacman turn: choose action with highest backed-up value
                return max(minimax(s, nextAgent, nextDepth) for s in successors)
            else:  # Ghost turn: assume adversarial response (lowest value)
                return min(minimax(s, nextAgent, nextDepth) for s in successors)

        # Root decision: choose the Pacman action with best minimax value.
        legalActions = gameState.getLegalActions(0)
        bestAction = max(
            legalActions,
            key=lambda a: minimax(
                gameState.generateSuccessor(0, a), 1, self.depth  # first ghost
            ),
        )
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
      

        def alphabeta(state, agentIndex, depth, alpha, beta):
            # Check if we've reached a terminal state or maximum depth
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # Get the list of moves the current agent is allowed to make.
            # If there are no legal actions, return the evaluation of the state
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            # Cycling through agents: after the last ghost moves, it's Pacman's turn again
            # Depth decrements only when it is Pacman's turn again
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # Pacman's turn: maximize value, alpha starts at -infinity
            if agentIndex == 0:
                value = float("-inf")
                
                # Loop through legal actions and generate successor states
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    
                    # recurse on the successor state to get its minimax value to update alpha and value
                    value = max(
                        value,
                        alphabeta(successor, nextAgent, nextDepth, alpha, beta),
                    )
                    
                    # Update alpha to the best value for Pacman so far
                    alpha = max(alpha, value)
                    # Project autograders expect no pruning on equality.
                    if beta < alpha:
                        break
                return value

            value = float("inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(
                    value,
                    alphabeta(successor, nextAgent, nextDepth, alpha, beta),
                )
                beta = min(beta, value)
                # Project autograders expect no pruning on equality.
                if beta < alpha:
                    break
            return value

        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in legalActions:
            value = alphabeta(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
