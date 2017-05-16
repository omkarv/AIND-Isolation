"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def test_initial_board(self):
        self.assertEqual(len(self.game.get_blank_spaces()), 49)

    def test_initial_legal_moves(self):
        self.assertEqual(len(self.game.get_legal_moves(self.player1)), 49)

    def test_player_initialized(self):
        self.assertEqual(self.game.active_player, "Player1")

    def test_minimax(self):
        isolation_player = game_agent.MinimaxPlayer()
        self.assertEqual(isolation_player.minimax(self.game, 2), (0, 0))

    def test_alphabeta(self):
        isolation_player = game_agent.AlphaBetaPlayer()
        self.assertEqual(isolation_player.alphabeta(self.game, 7), (0, 0))

    # def test_alphabeta_no_initial_depth(self):
    #     isolation_player = game_agent.AlphaBetaPlayer()
    #     self.assertEqual(isolation_player.alphabeta(self.game, 3), (0, 0))

if __name__ == '__main__':
    unittest.main()
