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

                    # If alpha is greater than beta - prune
                    #   This is because the minimax would never let the decision
                    #   reach this point because the best choice is accounted for.
                    # This implementation does not prune on eqauality for the
                    #   sake of the autograder.
                    if beta < alpha:
                        break
                # Returning the best value found.
                return value

            # Ghost behavior (minimizing branch)
            #   Same as maximizing behavior but beta will start at +infinity
            #   We update beta for min each time and prune when Beta is below
            #   alpha.
            value = float("inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(
                    value,
                    alphabeta(successor, nextAgent, nextDepth, alpha, beta),
                )
                beta = min(beta, value)
                # no prune on equality again because I want that autograder grade
                if beta < alpha:
                    break
            return value

        # Set up: default to the first lgal action, initializing tracking value
        #   to -infinity, and start alpha and beta at their bounds
        legalActions = gameState.getLegalActions(0)
        bestAction = legalActions[0]
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        # For each pacman action, generate the sucessor and get
        #   The alpha-beta value starting with the first ghost + full depth
        #   This loop threads alpha through siblings, meaning alpha represents
        #   the best score for each action possible and that is carried while we
        #   explore the other actions.
        for action in legalActions:
            value = alphabeta(
                gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta
            )
            # If this action is better than what we had, store. Then raise
            #   aloha so the next sibling's recursion can prune if needed.
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        # returns the best action found.
        return bestAction


# Difference compared to alpha-beta:
#  No pruning, because there is no alpha beta in general.
#  Assumption that adversaries (ghosts) are random, so we take the average
#  value of all children
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

        def expectimax(state, agentIndex, depth):
            # Terminal states and depth cutoff use the evaluation function.
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)

            # get this agent's moves: if there are none, return the evaluation of the state
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            # Cycle through agents
            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth - 1 if nextAgent == 0 else depth

            # Recursively get the expectimax value of each successor state after
            #   taking each legal action
            values = [
                expectimax(
                    state.generateSuccessor(agentIndex, action),
                    nextAgent,
                    nextDepth,
                )
                for action in legalActions
            ]

            # Pacman maximizes, return best value
            if agentIndex == 0:
                return max(values)

            # Ghosts: return average of children's values.
            #     Assuming ghost is random, each action has a 1/len(val) chance
            return sum(values) / float(len(values))

        # Return the action with the highest expectimax value, starting with the
        #   first ghost and full depth.
        legalActions = gameState.getLegalActions(0)
        return max(
            legalActions,
            key=lambda action: expectimax(
                gameState.generateSuccessor(0, action),
                1,
                self.depth,
            ),
        )



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Combines the true game score with state features that encourage
    Pacman to finish remaining food, move toward useful pellets/capsules, avoid
    active ghosts, and chase scared ghosts when they are close enough to catch.
    """
    
    # Prefer path to winning, avoid path to losing.
    if currentGameState.isWin():
        return 1000000.0 + currentGameState.getScore()
    if currentGameState.isLose():
        return -1000000.0 + currentGameState.getScore()

    # State info
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    # Base score from the game state, which accounts for food eaten, time penalties, etc.
    score = currentGameState.getScore()

    # Fewer remaining objectives is better, especially late game.
    score -= 8.0 * len(foodList)
    score -= 20.0 * len(capsules)

    # Using reciprocal distance to nearest food.
    # + 1 to avoid div by zero when on food
    if foodList:
        foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
        closestFood = min(foodDistances)
        score += 12.0 / (closestFood + 1)

        # penalty for fodo that's far away
        # Encourage pacman to finish food before going for further clusters.
        farthestFood = max(foodDistances)
        score -= 1.5 * farthestFood

    # Higher (better) score for being closer to powerup, because ghosts 
    #   Can be consumed then and safety timer.
    if capsules:
        closestCapsule = min(
            manhattanDistance(pacmanPos, capsule) for capsule in capsules
        )
        score += 10.0 / (closestCapsule + 1)

    # Ghosts: if scared, chase when close enough; if active, stay away.
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        distance = manhattanDistance(pacmanPos, ghostPos)

        if ghostState.scaredTimer > 0:
            if distance <= ghostState.scaredTimer:
                score += 5.0 / (distance + 1)
            else:
                score += 4.0 / (distance + 1)
        else:
            if distance <= 1:
                score -= 1000.0
            elif distance == 2:
                score -= 40.0
            else:
                score -= 8.0 / distance

    return score


# Abbreviation
better = betterEvaluationFunction
