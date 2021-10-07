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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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

        "*** YOUR CODE HERE ***"


        newFoodList = newFood.asList()
        capsuleList = successorGameState.getCapsules()
        score = 0

        action_score = {Directions.SOUTH:0,
                        Directions.NORTH:0,
                        Directions.EAST:0,
                        Directions.WEST:0,
                        Directions.STOP:-100000}

        if newFood.count() == currentGameState.getFood().count():
            mindist = 2**64
            for foodPos in newFoodList:
                mindist = min(mindist, util.manhattanDistance(foodPos, newPos))
            score += 100.0/(mindist*min(3,len(newFoodList))) + action_score[action]
        else:
            score = 104

        if len(capsuleList) == len(currentGameState.getCapsules()):
            mindist = 2**64
            for capsulePos in capsuleList:
                mindist = min(mindist, util.manhattanDistance(capsulePos, newPos))
            score += 10000.0/(mindist*min(3,len(capsuleList)+1)) 
        else:
            score += 10009
        

        ghostScore = 0
        for ghostState in newGhostStates:
            dist = util.manhattanDistance(ghostState.getPosition(), newPos)
            scared = ghostState.scaredTimer
            if scared==0:
                if dist > 2:
                    continue
                ghostScore += 12**(3 - dist)
            else:
                if scared <= 2 and dist<=2:
                    ghostScore += 10000000/dist
                else:
                    ghostScore -= 10000000/dist
            if dist==0:
                print(dist)
                  
        
        score -= ghostScore

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    '''
    def graphExpansion(self, depth, currAgent, numAgents, gameState):
        if depth == self.depth:
            print(self.evaluationFunction(gameState))
            return self.evaluationFunction(gameState)

        if currAgent > numAgents:
            return self.graphExpansion(depth+1, 0, numAgents, gameState)

        if currAgent == 0:
            maxScore = 0
            for action in gameState.getLegalActions(0):
                next = gameState.generateSuccessor(0, action)
                if next.isLose():
                    continue
                score = self.graphExpansion(depth, currAgent+1, numAgents, next)
                if maxScore < score:
                    maxScore = score
            return maxScore
        else:
            minScore = 99999
            for action in gameState.getLegalActions(0):
                next = gameState.generateSuccessor(0, action)
                score = self.graphExpansion(depth, currAgent+1, numAgents, next)
                if minScore == None or minScore > score:
                    minScore = score
            return minScore
    '''

    def maxValue(self, gameState, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return score, None


        maxScore, bestAction = -2**64, Directions.STOP
        for action in gameState.getLegalActions():
            newGameState = gameState.generateSuccessor(0, action)
            curScore = self.minValue(newGameState, 1, depth)
            if maxScore < curScore:
                maxScore = curScore
                bestAction = action

        return maxScore, bestAction


    def minValue(self, gameState, ghost, depth):
        if gameState.isWin() or gameState.isLose() or ghost>=gameState.getNumAgents():
            return self.evaluationFunction(gameState)

        minScore = 2**63
        validActions = gameState.getLegalActions(ghost)

        for action in validActions:
            nextGameState = gameState.generateSuccessor(ghost, action)
            if ghost==gameState.getNumAgents()-1:
                if depth==1:
                    curScore = self.evaluationFunction(nextGameState)
                else:
                    curScore, _ = self.maxValue(nextGameState, depth-1)
            else:
                curScore = self.minValue(nextGameState, ghost+1, depth)
            
            if curScore < minScore:
                minScore = curScore

        return minScore

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
        "*** YOUR CODE HERE ***"
        '''
            maxScore = 0
            maxAction = None
    
            for action in gameState.getLegalActions(0):
                print("enter")
                score = self.graphExpansion(0, 1, gameState.getNumAgents(), gameState.generateSuccessor(0, action))
                print(score)
                if score >= maxScore:
                    maxAction = action
                    maxScore = score
    
            print(maxScore)
    
            return maxAction
        '''

        score, action = self.maxValue(gameState, self.depth)

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, gameState, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            score = self.evaluationFunction(gameState)
            return score, None
        
        maxScore, bestAction = -2**128, None
        for action in gameState.getLegalActions():
            newGameState = gameState.generateSuccessor(0, action)
            curScore = self.expValue(newGameState, 1, depth)
            if maxScore <= curScore:
                maxScore = curScore
                bestAction = action
        
        return maxScore, bestAction


    def expValue(self, gameState, ghost, depth):
        if gameState.isWin() or gameState.isLose() or ghost==gameState.getNumAgents():
            return self.evaluationFunction(gameState)

        expectedScore = 0
        validActions = gameState.getLegalActions(ghost)

        for action in validActions:
            nextGameState = gameState.generateSuccessor(ghost, action)
            if ghost==gameState.getNumAgents()-1:
                if depth==1:
                    curScore = self.evaluationFunction(nextGameState)
                else:
                    curScore, _ = self.maxValue(nextGameState, depth-1)
            else:
                curScore = self.expValue(nextGameState, ghost+1, depth)
            expectedScore += curScore/len(validActions)
        
        return expectedScore




    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxScore, action = self.maxValue(gameState, self.depth)
        return action
        

def bfs(currentGameState):
    q = util.Queue()
    x,y = currentGameState.getPacmanPosition()
    q.push((x,y))
    walls = currentGameState.getWalls()
    n, m = walls.height+1, walls.width+1

    inf, mindistToFood = 1e9, 1e9
    dist = [[inf for j in range(n)] for i in range(m)]
    dist[x][y] = 0

    delta = [(0,1), (1,0), (0,-1), (-1,0)]

    while not q.isEmpty():
        curx, cury = q.pop()
        if currentGameState.hasFood(curx, cury):
            mindistToFood = min(mindistToFood, dist[curx][cury])
        for (dx, dy) in delta:
            nextx, nexty = curx+dx, cury+dy
            if walls[nextx][nexty] == False:
                if dist[nextx][nexty] == inf:
                    dist[nextx][nexty] = dist[curx][cury]+1
                    q.push((nextx, nexty))

    return mindistToFood, dist


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    inf = 1e9
    mindistToFood, dist = bfs(currentGameState)
    if currentGameState.isWin():
        score = inf
    elif currentGameState.isLose():
        score = -2**50
    else:
        score = 1/mindistToFood - currentGameState.getFood().count()

    capsuleList = currentGameState.getCapsules()
    
    mindistToCapsule = inf
    for (x,y) in capsuleList:
        mindistToCapsule = min(mindistToCapsule, dist[x][y])
    
    # score += 3/(len(capsuleList)+1)

    ghostScore = 0
    ghostNum = len(currentGameState.getGhostStates())
    for ghostState in currentGameState.getGhostStates():
        ghostPos = ghostState.getPosition()
        distGhost = dist[int(ghostPos[0])][int(ghostPos[1])]
        scared = ghostState.scaredTimer
        
        if scared > distGhost and distGhost<20:
            ghostScore -= 10000/(distGhost+1) 

    score -= ghostScore

    return  score - currentGameState.getFood().count() 

# Abbreviation
better = betterEvaluationFunction
