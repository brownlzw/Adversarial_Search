from abc import ABCMeta, abstractmethod

###############################################################################
# An AdversarialSearchProblem is a representation of a game that is convenient
# for running adversarial search algorithms.
#
# A game can be put into this form by extending the AdversarialSearchProblem
# class. See tttproblem.py for an example of this.
#
# Every subclass of AdversarialSearchProblem has its game states represented
# as instances of a subclass of GameState. The only requirement that of a
# subclass of GameState is that it must implement that player_to_move(.) method,
# which returns the index (0-indexed) of the next player to move.
###############################################################################

class GameState:
	__metaclass__ = ABCMeta

	@abstractmethod
	def player_to_move(self):
		"""
		Produces the index of the player who will move next.
		"""
		pass

class AdversarialSearchProblem:
	__metaclass__ = ABCMeta

	def get_start_state(self):
		"""
		Produces the state from which to start.
		"""
		return self._start_state

	def set_start_state(self, state):
		"""
		Changes the start state to the given state.
		Note to student: You should not need to use this.
		This is only for running games.
		"""
		self._start_state = state

	@abstractmethod
	def get_available_actions(self, state):
		"""
		Returns the set of actions available to the player-to-move
		from the current state.
		"""
		pass

	@abstractmethod
	def transition(self, state, action):
		"""
		Returns the state that results from taking the given action
		from the given state. (Assume deterministic transitions.)
		"""
		assert not(self.is_terminal_state(state))
		assert action in self.get_available_actions(state)
		pass

	@abstractmethod
	def is_terminal_state(self, state):
		"""
		Produces a boolean indicating whether or not the given state is terminal.
		"""
		pass

	@abstractmethod
	def evaluate_state(self, state):
		"""
		Takes in a terminal state and produces a list of numbers
		such that the i'th entry represents how good the state
		is for player i.
		"""
		assert self.is_terminal_state(state)
		pass