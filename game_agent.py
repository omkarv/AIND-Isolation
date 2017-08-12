"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_score = 0
    if game.get_player_location(player) is not None:
        y, x = game.get_player_location(player)
        w, h = game.width / 2., game.height / 2.
        center_score = float((h - y)**2 + (w - x)**2)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - opp_moves - 0.5 * math.sqrt(center_score))

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves + len(game.get_blank_spaces()))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_score = 0
    if game.get_player_location(player) is not None:
        y, x = game.get_player_location(player)
        w, h = game.width / 2., game.height / 2.
        center_score = float((h - y)**2 + (w - x)**2)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(own_moves - opp_moves - math.sqrt(center_score))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def terminal_test(self, current_depth, moves):
        return current_depth > self.search_depth or len(moves) is 0

    def terminal_test_return_min_player(self, game):
        # ensure we return score from the perspective of the maximizing player
        return self.score(game, game.get_opponent(game.active_player))

    def terminal_test_return_max_player(self, game):
        return self.score(game, game.active_player)

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            # select any legal move to prevent forfeits
            moves = game.get_legal_moves()
            if not moves:
                best_move = moves[0]

        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        def min_value(self, game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            moves = game.get_legal_moves()

            if (self.terminal_test(current_depth, moves)):
                return self.terminal_test_return_min_player(game)

            min_utility = float("inf")

            for move in moves:
                move_utility = max_value(self, game.forecast_move(move), current_depth + 1)

                if (move_utility < min_utility):
                    min_utility = move_utility

            return min_utility

        def max_value(self, game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            moves = game.get_legal_moves(game.active_player)

            if self.terminal_test(current_depth, moves):
                return self.terminal_test_return_max_player(game)

            max_utility = float("-inf")

            for move in moves:
                move_utility = min_value(self, game.forecast_move(move), current_depth + 1)

                if (move_utility > max_utility):
                    max_utility = move_utility

            return max_utility

        # initiailize minimax routine
        moves = game.get_legal_moves()
        max_utility = float("-inf")
        best_move = (-1, -1)

        if (len(moves) > 0):
            best_move = moves[0]

        # main minimax routine
        for move in moves:
            # start at depth level 2, since we are effectively at level 1 in
            # this loop
            move_utility = min_value(self, game.forecast_move(move), 2)

            if move_utility > max_utility:
                max_utility = move_utility
                best_move = move

        return best_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def terminal_test(self, depth, moves):
        return depth <= 0 or len(moves) is 0

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        best_move = (-1, -1)
        depth = 0

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                depth = depth + 1
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            # select any move if search timeout
            moves = game.get_legal_moves()
            if (len(moves) > 0 and best_move is (-1, -1)):
                best_move = moves[random.randint(0, len(moves) - 1)]

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers


        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = (-1, -1)

        moves = game.get_legal_moves()
        if len(moves) > 0:
            # select a move that doesn't forfeit the game
            best_move = moves[random.randint(0, len(moves) - 1)]

        def max_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            
            if self.terminal_test(depth, moves):
                return self.terminal_test_return_max_player(game)

            # the notation for utility is 'v' in AIMA https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
            utility = float("-inf")
            for move in game.get_legal_moves():
                alphabeta_value = min_value(self, game.forecast_move(move), depth - 1, alpha, beta)

                if (alphabeta_value > utility):
                    utility = alphabeta_value

                if (utility >= beta):
                    return utility

                alpha = max(alpha, utility)
            return utility

        def min_value(self, game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if self.terminal_test(depth, moves):
                return self.terminal_test_return_min_player(game)

            utility = float("inf")
            for move in game.get_legal_moves():
                alphabeta_value = max_value(self, game.forecast_move(move), depth - 1, alpha, beta)

                # only set utility to alphabeta_value, if alphabeta_value is the best minimum so far
                if (alphabeta_value < utility):
                    utility = alphabeta_value

                if (utility <= alpha):
                    return utility

                beta = min(beta, utility)

            return utility

        beta = float("inf")
        alpha = float("-inf")
        max_utility = float("-inf")
        # main alphabeta routine
        for move in moves:

            move_utility = min_value(self, game.forecast_move(move), depth - 1, alpha, beta)

            if move_utility > max_utility:
                max_utility = move_utility
                best_move = move

            alpha = max(alpha, move_utility)

        return best_move
