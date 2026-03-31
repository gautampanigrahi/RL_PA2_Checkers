import functools
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, Box, Tuple
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector, wrappers


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {"render_modes": ["human"], "name": "checkers_6x6"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["player_1", "player_2"]
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-2, high=2, shape=(6, 6), dtype=np.int8)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Tuple((Discrete(6), Discrete(6), Discrete(6), Discrete(6)))

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        symbols = {
            0: ".",
            1: "r",
            2: "R",
            -1: "b",
            -2: "B"
        }

        print()
        print("  0 1 2 3 4 5")
        for r in range(6):
            print(f"{r} " + " ".join(symbols[self.board[r][c]] for c in range(6)))
        if self.last_agent is not None:
            print(f"Last move by: {self.last_agent}")
        #print(f"Next turn: {self.agent_selection}")

    def observe(self, agent):
        return np.array(self.board, dtype=np.int8)

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        self.board = [[0 for _ in range(6)] for _ in range(6)]
        for r in range(2):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r][c] = 1
        for r in range(4, 6):
            for c in range(6):
                if (r + c) % 2 == 1:
                    self.board[r][c] = -1

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        self.must_continue_capture = False
        self.forced_piece = None
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.infos[self.agent_selection]["legal_moves"] = self._get_legal_moves(self.agent_selection)       #populating legal moves for the first agent immediately after game reset
        self.last_agent = None

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        
        self.last_agent = self.agent_selection
        agent = self.agent_selection
        opponent = "player_2" if agent == "player_1" else "player_1"
        self._cumulative_rewards[agent] = 0
        self.rewards = {a: 0 for a in self.agents}
        legal_moves = self._get_legal_moves(agent)
        if action not in legal_moves:
            raise ValueError(f"Illegal action {action} for {agent}. Legal moves: {legal_moves}")

        from_r, from_c, to_r, to_c = action
        piece = self.board[from_r][from_c]
        self.board[from_r][from_c] = 0
        self.board[to_r][to_c] = piece

        was_capture = False
        if abs(to_r - from_r) == 2:
            mid_r = (from_r + to_r) // 2
            mid_c = (from_c + to_c) // 2
            self.board[mid_r][mid_c] = 0
            was_capture = True

        #promotion to king
        if piece == 1 and to_r == 5:
            self.board[to_r][to_c] = 2
        elif piece == -1 and to_r == 0:
            self.board[to_r][to_c] = -2

        #Checking draw(if both the players only have kings)
        if self._only_kings_left():
            self.terminations = {a: True for a in self.agents}
            self.rewards["player_1"] = 0.5
            self.rewards["player_2"] = 0.5
            self.infos["player_1"]["winner"] = "draw"
            self.infos["player_2"]["winner"] = "draw"

            self._accumulate_rewards()

            if self.render_mode == "human":
                self.render()
            return

        #if piece was captured assign 0.25 reward to agent
        if was_capture:
            self.rewards[agent] = 0.25
            #checking to see if more pieces can be captured ie.chain capture
            more_captures = self._get_piece_captures(to_r, to_c)
            if more_captures:
                self.must_continue_capture = True
                self.forced_piece = (to_r, to_c)
                self.agent_selection = agent
                self.infos[agent]["legal_moves"] = more_captures
                self._accumulate_rewards()

                if self.render_mode == "human":
                    self.render()
                return

        self.must_continue_capture = False
        self.forced_piece = None

        opponent_has_piece = False
        for r in range(6):
            for c in range(6):
                p = self.board[r][c]
                if opponent == "player_1" and p in (1, 2):
                    opponent_has_piece = True
                elif opponent == "player_2" and p in (-1, -2):
                    opponent_has_piece = True

        opponent_legal_moves = self._get_legal_moves(opponent)

        # opponent does not have any legal moves or any pieces so terminate 
        if (not opponent_has_piece) or (len(opponent_legal_moves) == 0):
            self.terminations = {a: True for a in self.agents}
            self.rewards[agent] = 1
            self.rewards[opponent] = -1
            self.infos[agent]["winner"] = agent
            self.infos[opponent]["winner"] = agent
        else:
            self.agent_selection = opponent
            self.infos[opponent]["legal_moves"] = opponent_legal_moves

        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def _get_piece_captures(self, r, c):
        piece = self.board[r][c]
        if piece == 0:
            return []

        piece_sign = 1 if piece > 0 else -1
        directions = self._get_directions(piece_sign, abs(piece) == 2)
        moves = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            jr, jc = r + 2 * dr, c + 2 * dc

            if self._in_bounds(nr, nc) and self._in_bounds(jr, jc):
                if self._is_enemy_piece(self.board[nr][nc], piece_sign) and self.board[jr][jc] == 0:
                    moves.append((r, c, jr, jc))

        return moves

    def _get_legal_moves(self, agent):
        if self.must_continue_capture and self.forced_piece is not None:
            r, c = self.forced_piece
            return self._get_piece_captures(r, c)

        piece_sign = 1 if agent == "player_1" else -1
        legal_moves = []
        capture_moves = []

        for r in range(6):
            for c in range(6):
                piece = self.board[r][c]

                if piece_sign == 1 and piece not in (1, 2):
                    continue
                if piece_sign == -1 and piece not in (-1, -2):
                    continue

                is_king = abs(piece) == 2
                directions = self._get_directions(piece_sign, is_king)

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    jr, jc = r + 2 * dr, c + 2 * dc

                    if self._in_bounds(nr, nc) and self.board[nr][nc] == 0:
                        legal_moves.append((r, c, nr, nc))

                    if self._in_bounds(nr, nc) and self._in_bounds(jr, jc):
                        middle_piece = self.board[nr][nc]
                        landing_piece = self.board[jr][jc]

                        if self._is_enemy_piece(middle_piece, piece_sign) and landing_piece == 0:
                            capture_moves.append((r, c, jr, jc))

        return capture_moves if capture_moves else legal_moves

    def _get_directions(self, piece_sign, is_king):
        if is_king:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if piece_sign == 1:
            return [(1, -1), (1, 1)]
        return [(-1, -1), (-1, 1)]

    def _in_bounds(self, r, c):
        return 0 <= r < 6 and 0 <= c < 6

    def _is_enemy_piece(self, piece, piece_sign):
        if piece_sign == 1:
            return piece in (-1, -2)
        return piece in (1, 2)
    
    def _only_kings_left(self):
        p1_has_piece = False
        p2_has_piece = False

        for r in range(6):
            for c in range(6):
                piece = self.board[r][c]
                if piece == 1:      
                    return False
                elif piece == -1:   
                    return False
                elif piece == 2:
                    p1_has_piece = True
                elif piece == -2:
                    p2_has_piece = True

        return p1_has_piece and p2_has_piece