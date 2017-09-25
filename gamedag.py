from adversarialsearchproblem import AdversarialSearchProblem, GameState

class DAGState(GameState):
    def __init__(self,index,player_to_move):
        self._index = index
        self._ptm = player_to_move
        
    def player_to_move(self):
        return self._ptm

class GameDAG(AdversarialSearchProblem):
    def __init__(self, matrix, start_state, terminal_indices, evaluations_at_terminal,turns):
        """
        Inputs:
        matrix - an n-by-n 2D array storing booleans, where n is 
        the number of states in the game.

        Each state corresponds to a unique natural number between 0 and n-1 inclusive,
        such that it corresponds to an index in the array. 
        matrix[i][j] stores the proposition that it is possible to transition 
        from the state whose index is i to the state whose index is j.

        start_state - a DAGState representing a starting state

        terminal_indices - a set of natural numbers that correspond to the
        terminal states of the game

        evaluations_at_terminal - a dictionary where the key is the terminal state
        and the value is a list of evaluations per player at that terminal state
        
        turns - a list whose indices correspond to states and whose values are 
        the player that moves in that state
        """
        self._matrix = matrix
        self._start_state = start_state
        self._terminal_indices = terminal_indices
        self._evaluations_at_terminal = evaluations_at_terminal
        self._turns = turns

    def get_available_actions(self, state):
        """
        Returns the set of next states (actions) available to the player-to-move
        from the current state.
        Returns an empty set if the state is a terminal state.
        """
        if self.is_terminal_state(state):
            return set()
        available_actions = set()
        actions_to_check = self._matrix[state._index]
        for i in range (0, len(actions_to_check)):
            if actions_to_check[i]:
                available_actions.add(i)
        return available_actions
  
    def transition(self, state, action):
        """
        Returns the state that results from taking the given action
        from the given state. (Assume deterministic transitions.)
        """
        assert not(self.is_terminal_state(state))
        assert action in self.get_available_actions(state)
        
        return DAGState(action,self._turns[action])

    def is_terminal_state(self, state):
        """
        Returns a boolean indicating whether or not the given state is terminal.
        """
        return state._index in self._terminal_indices

    def evaluate_state(self, state):
        """
        Takes in a nonterminal state, and, if the state is terminal,
        produces a list of nonnegative numbers that sum to 1, where the i'th
        element of the list is the value of the state to player i.
        Most commonly, this list will have a 1 at some index j, and all 0's
        everywhere else, indicating that player j won the game.
        """
        assert self.is_terminal_state(state)

        return self._evaluations_at_terminal[state._index]