# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
from _subprocess import INFINITE
from pacman import GameState


BIGNUM = 10000
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

    Just like in the previous project, getAction takes a gameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
    
    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentgameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    gameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a gameState (pacman.py)
    successorgameState = currentgameState.generatePacmanSuccessor(action)
    newPos = successorgameState.getPacmanPosition()
    newFood = successorgameState.getFood()
    newGhostStates = successorgameState.getGhostStates()
    foodNum = currentgameState.getFood().count()
    if len(newFood.asList()) == foodNum:  # if this action does not eat a food 
        dis = BIGNUM
        for pt in newFood.asList():
            if manhattanDistance(pt , newPos) < dis :
                dis = manhattanDistance(pt, newPos)
    else:
        dis = 0
    for ghost in newGhostStates:  # the impact of ghost surges as distance get close
        dis += 4 ** (2 - manhattanDistance(ghost.getPosition(), newPos))
    return -dis

def scoreEvaluationFunction(currentgameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentgameState.getScore()

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

  def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
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
    
      Directions.STOP:
        The stop direction, which is always legal
    
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
    
      gameState.getNumAgents():
        Returns the total number of agents in the game
      """
    
      if gameState.isWin() or gameState.isLose():
          return Directions.STOP
      nextMoves = gameState.getLegalPacmanActions()
      num = gameState.getNumAgents() - 1
      value = -BIGNUM
      chosenMove = Directions.STOP
      for move in nextMoves:
          nextState = gameState.generatePacmanSuccessor(move)
          if nextState.isWin():
              return move  # win the game immediately if it can 
          score = self.moveGhost(nextState, num, 1)
          if score > value:
              value = score
              chosenMove = move
      return chosenMove
  
  def moveAgent(self, gameState , depth):
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      nextMoves = gameState.getLegalPacmanActions()
      nextStates = [gameState.generatePacmanSuccessor(action) for action in nextMoves]
      num = gameState.getNumAgents() - 1
      scores = [self.moveGhost(nextState , num , depth + 1)  for nextState in nextStates]
      return max(scores)
    
  def moveGhost(self , gameState , ghostNum , depth):
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      nextMoves = gameState.getLegalActions(ghostNum)
      if len(nextMoves) == 0:
          return self.evaluationFunction(gameState)
      nextStates = [gameState.generateSuccessor(ghostNum, action)  for action in nextMoves]
      num = ghostNum - 1
      if num == 0:  # all ghosts has been moved 
          if depth == self.depth:  # has already explored enough depth 
              scores = [self.evaluationFunction(nextState) for nextState in nextStates]
          else:  # explore deeper, make pacman move the next 
              scores = [self.moveAgent(nextState , depth) for nextState in nextStates]
      else:  # move the next ghost in the list
          scores = [self.moveGhost(nextState, num , depth) for nextState in nextStates]
      return min(scores)
  
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.
    
      Here are some method calls that might be useful when implementing minimax.
    
      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
    
      Directions.STOP:
        The stop direction, which is always legal
    
      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
    
      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    if gameState.isWin() or gameState.isLose():
        return Directions.STOP
    nextMoves = gameState.getLegalPacmanActions()
    num = gameState.getNumAgents() - 1
    value = -BIGNUM
    chosenMove = Directions.STOP
    for move in nextMoves:
        nextState = gameState.generatePacmanSuccessor(move)
        if nextState.isWin():
            return move  # win the game immediately if it can 
        score = self.moveGhost(nextState, value , INFINITE, num, 1)
        if score > value:
            value = score
            chosenMove = move
    return chosenMove
    
  def moveAgent(self, gameState , alpha , beta, depth):
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      nextMoves = gameState.getLegalPacmanActions()
      num = gameState.getNumAgents() - 1
      value = -BIGNUM
      for move in nextMoves:
          nextState = gameState.generatePacmanSuccessor(move)
          score = self.moveGhost(nextState, alpha , beta, num, depth + 1)
          value = max(value , score)
          if value >= beta:
              return value
          alpha = max(alpha , value)
      return value
    
  def moveGhost(self , gameState , alpha , beta , ghostNum , depth):
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      nextMoves = gameState.getLegalActions(ghostNum)
      num = ghostNum - 1
      value = BIGNUM
      for move in nextMoves:
          nextState = gameState.generateSuccessor(ghostNum , move)
          if num == 0:
              if depth == self.depth:
                  score = self.evaluationFunction(nextState)
              else:
                  score = self.moveAgent(nextState, alpha, beta, depth)
          else:
              score = self.moveGhost(nextState, alpha , beta, num, depth)
          value = min(value , score)
          if value <= alpha:
              return value
          beta = min(beta , value)
      return value
  
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
    if gameState.isWin() or gameState.isLose():
        return Directions.STOP
    nextMoves = gameState.getLegalPacmanActions()
    num = gameState.getNumAgents() - 1
    value = -BIGNUM
    chosenMove = Directions.STOP
    for move in nextMoves:
        nextState = gameState.generatePacmanSuccessor(move)
        if nextState.isWin():
            return move  # win the game immediately if it can 
        score = self.moveGhost(nextState, num, 1)
        if score > value:
            value = score
            chosenMove = move
    return chosenMove

  def moveAgent(self, gameState , depth):
      if gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)
      nextMoves = gameState.getLegalPacmanActions()
      nextStates = [gameState.generatePacmanSuccessor(action) for action in nextMoves]
      num = gameState.getNumAgents() - 1
      scores = [self.moveGhost(nextState , num , depth + 1)  for nextState in nextStates]
      return max(scores)  # return the best choice of pacman
    
  def moveGhost(self , gameState , ghostNum , depth):
    if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    nextMoves = gameState.getLegalActions(ghostNum)
    num = ghostNum - 1
    nextStates = [ gameState.generateSuccessor(ghostNum , move) for move in nextMoves]
    if num == 0:
        if depth == self.depth:
            scores = [self.evaluationFunction(nextState) for nextState in  nextStates]
        else:
            scores = [ self.moveAgent(nextState , depth) for nextState in nextStates]
    else:
        scores = [self.moveGhost(nextState, num, depth) for nextState in nextStates]
    return sum(scores) / len(scores)
  
def betterEvaluationFunction(currentgameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      
      Without any disturbance of ghost, the pacman would go after the closest food, similar to
      ClosestDotSearchAgent in the previous project. The implementation involves a simple
      breath-first-searce that terminates at the closest food near pacman.
      
      The effect of ghost on a pacman surges drastically when they are close (one block away). 
      Away from pacman, ghosts have little effect. The implementation involves an exponential 
      function.
      
      To take advantage of the pellet, every scared ghost will pose no negative influence, but
      on the contrary, add 25 bonus points. So the more active ghosts that are chasing after the
      pacman, the more it wants to take a pellet. 
      At the very beginning of the experiment game, there are three ghosts and the pellet is very
      close, so the pacman is very likely to use one pellet before proceed to clear the food.
      However, after taking one pellet, it creates no marginal benefit to take another one, so the
      pacman go directly after the closest food. 
      Sometimes the pacman would avoid a scared ghost, because eating it produces an active ghost
      in the center and lose the bonus points, so if there is a choice, it would tend to leave
      scared ghosts alone and take as much advantage of the scared time as possible. However, this
      tendency to avoid scared ghosts is overriden by the desire to eat food, since the bonus of
      food 30 is larger that of a scared ghost, so scared ghost would not be in the way to clear
      food.
      
    """
    if currentgameState.isWin():
        return BIGNUM
    elif currentgameState.isLose():
        return -BIGNUM
    pos = currentgameState.getPacmanPosition()
    food = currentgameState.getFood()
    walls = currentgameState.getWalls()
    dmap = walls.copy()
    stk = util.Queue()
    stk.push(pos)
    dmap[pos[0]][pos[1]] = 0
    dis = 0
    while not stk.isEmpty():  # use BFS to aim at the closest food if not disturbed
        x , y = stk.pop()
        dis = dmap[x][y] + 1
        if food[x][y]:
            break;
        for v in [(0, 1) , (1, 0) , (0 , -1) , (-1 , 0)]:
            xn = x + v[0]
            yn = y + v[1]
            if dmap[xn][yn] == False:
                dmap[xn][yn] = dis
                stk.push((xn, yn))
    ret = 1 - dis
    ghosts = currentgameState.getGhostStates()
    for ghost in ghosts:
        if ghost.scaredTimer == 0:  # active ghost poses danger to the pacman
            ret -= 100 ** (1.6 - manhattanDistance(ghost.getPosition(), pos))
        else:  # bonus points for having a scared ghost 
            ret += 25
    ret -= 30 * food.count()  # bonus points for eating a food 
    return ret
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(ExpectimaxAgent):
    """
      Your agent for the mini-contest
      I simply use the ExpectimaxAgent above, which is able to make the best use of pellet against 
      ghosts and chase down the closest food through the map with BFS.
      
      Please read the documentation in ExpectimaxAgent
    """
    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
      self.index = 0  # Pacman is always agent index 0
      self.evaluationFunction = util.lookup(evalFn, globals())
      self.depth = int(depth)
