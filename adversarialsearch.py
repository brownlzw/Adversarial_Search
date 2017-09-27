from adversarialsearchproblem import AdversarialSearchProblem
from sets import Set
from gamedag import GameDAG, DAGState

# INPUT SPEC: All functions below take in an AdversarialSearchProblem (ASP)

# OUTPUT SPEC: All functions below output an action. More specifically,
# they output an element of asp.get_available_actions(asp.get_start_state())
def minimax(asp):
	"""
	Implement the minimax algorithm on ASPs,
	assuming that the given game is both 2-player and constant-sum
	"""
	def min_value(state):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = float("inf")
		for act in asp.get_available_actions(state):
			val = min(val, max_value(asp.transition(state, act)))
		return val

	def max_value(state):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = -float("inf")
		for act in asp.get_available_actions(state):
			val = max(val, min_value(asp.transition(state, act)))
		return val

	start = asp.get_start_state()
	player = start.player_to_move()
	if asp.is_terminal_state(start):
		return asp.evaluate_state(start)
	max_val = -float("inf")
	max_action = None
	for action in asp.get_available_actions(start):
		child = min_value(asp.transition(start, action))
		if child > max_val:
			max_val = child
			max_action = action
	return max_action


def alpha_beta(asp):
	"""
	Implement the alpha-beta pruning algorithm on ASPs,
	assuming that the given game is both 2-player and constant-sum.
	"""
	def min_value(state, alpha, beta):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = float("inf")
		for act in asp.get_available_actions(state):
			val = min(val, max_value(asp.transition(state, act), alpha, beta))
			if val <= alpha:
				return val
			beta = min(beta, val)
		return val

	def max_value(state, alpha, beta):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = -float("inf")
		for act in asp.get_available_actions(state):
			val = max(val, min_value(asp.transition(state, act), alpha, beta))
			if val >= beta:
				return val
			alpha = max(alpha, val)
		return val

	start = asp.get_start_state()
	player = start.player_to_move()
	if asp.is_terminal_state(start):
		return asp.evaluate_state(start)
	alpha = -float("inf")
	beta = float("inf")
	max_val = -float("inf")
	max_action = None
	for action in asp.get_available_actions(start):
		child_val = min_value(asp.transition(start, action), alpha, beta)
		if child_val > max_val:
			max_val = child_val
			max_action = action
		alpha = max(alpha, max_val)
	return max_action


def eval_func(state):
	if state == 1:
		return -2
	if state == 2:
		return -4
	if state == 3:
		return -1
	if state == 4:
		return -2
	if state == 5:
		return -3
	if state == 6:
		return -4
	else:
		return -5

def alpha_beta_cutoff(asp, cutoff_ply, eval_func):
	"""
	This function should:
	- search through the asp using alpha-beta pruning
	- cut off the search after cutoff_ply moves have been made.
	 For example, when cutoff_ply = 1, use eval_func to evaluate
	 states that result from your player move. When cutoff_ply = 2, use
	 eval_func to evaluate states that result from your opponent's player move.
	 When cutoff_ply = 3 use eval_func to evaluate the states that result from
	 your second move.
	You may assume that cutoff_ply > 0.
 
	eval_func is a function that takes in a GameState and outputs
	a real number indicating how good that state is for the
	player who is using alpha_beta_cutoff to choose their action.
	"""
	def min_value(state, alpha, beta, d):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		if d == cutoff_ply:
			return eval_func(state)
		val = float("inf")
		for act in asp.get_available_actions(state):
			val = min(val, max_value(asp.transition(state, act), alpha, beta, d + 1))
			if val <= alpha:
				return val
			beta = min(beta, val)
		return val

	def max_value(state, alpha, beta, d):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		if d == cutoff_ply:
			return eval_func(state)
		val = -float("inf")
		for act in asp.get_available_actions(state):
			val = max(val, min_value(asp.transition(state, act), alpha, beta, d + 1))
			if val >= beta:
				return val
			alpha = max(alpha, val)
		return val

	start = asp.get_start_state()
	player = start.player_to_move()
	if asp.is_terminal_state(start):
		return asp.evaluate_state(start)
	alpha = -float("inf")
	beta = float("inf")
	max_val = -float("inf")
	max_action = None
	for action in asp.get_available_actions(start):
		child_val = min_value(asp.transition(start, action), alpha, beta, 1)
		if child_val > max_val:
			max_val = child_val
			max_action = action
		alpha = max(alpha, max_val)
	return max_action

def general_minimax(asp):
	"""
	Implement the generalization of the minimax algorithm that was
	discussed in the handout, making no assumptions about the
	number of players or reward structure of the given game.
	"""
	def min_value(state):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = float("inf")
		for act in asp.get_available_actions(state):
			next_state = asp.transition(state, act)
			if next_state.player_to_move() == player:
				val = min(val, max_value(next_state))
			else:
				val = min(val, min_value(next_state))
		return val

	def max_value(state):
		if asp.is_terminal_state(state):
			return asp.evaluate_state(state)[player]
		val = -float("inf")
		for act in asp.get_available_actions(state):
			next_state = asp.transition(state, act)
			if next_state.player_to_move() == player:
				val = max(val, max_value(next_state))
			else:
				val = max(val, min_value(next_state))
		return val

	start = asp.get_start_state()
	player = start.player_to_move()
	if asp.is_terminal_state(start):
		return asp.evaluate_state(start)
	max_val = -float("inf")
	max_action = None
	for action in asp.get_available_actions(start):
		child_state = asp.transition(start, action)
		if child_state.player_to_move() == player:
			next_val = max_value(child_state)
		else:
			next_val = min_value(child_state)
		if next_val > max_val:
			max_val = next_val
			max_action = action
	return max_action


def example_dag():
	"""
	An example of an implemented GameDAG from the gamedag class.
	Look at handout in section 3.3 to see visualization of the tree.
	"""

	matrix = [[False, True, True, False, False, False, False],
			  [False, False, False, True, True, False, False],
			  [False, False, False, False, False, True, True],
			  [False, False, False, False, False, False, False],
			  [False, False, False, False, False, False, False],
			  [False, False, False, False, False, False, False],
			  [False, False, False, False, False, False, False]]
	start_state = DAGState(0,0)
	terminal_indices = Set([3, 4, 5, 6])
	evaluations_at_terminal = {3: [-1, 1], 4: [-2, 2], 5: [-3, 3], 6: [-4, 4]}
	turns = [0, 1, 1, 0, 0, 0, 0]
	dag = GameDAG(matrix, start_state, terminal_indices, evaluations_at_terminal,turns)
	return dag

print minimax(example_dag())
print alpha_beta(example_dag())
print alpha_beta_cutoff(example_dag(), 1, eval_func)
print general_minimax(example_dag())
