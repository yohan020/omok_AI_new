# 파일명: environment.py

import numpy as np

class OmokEnv:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1: 흑, -1: 백
        self.winner = 0

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        if self.current_player == 1:
            state[0] = (self.board == 1).astype(np.float32)
            state[1] = (self.board == -1).astype(np.float32)
        else:
            state[0] = (self.board == -1).astype(np.float32)
            state[1] = (self.board == 1).astype(np.float32)
        return state

    def get_valid_moves(self):
        return (self.board.flatten() == 0).astype(int)

    def check_win(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else: break
            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else: break
            if count >= 5: return True
        return False

    # (!!!) 추가된 기능: 공격적인 패턴(열린3, 열린4) 감지
    def check_attack_pattern(self, row, col, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        bonus_reward = 0.0
        
        # 4목: 거의 승리와 같으므로 높은 점수
        # 3목: 좋은 공격 찬스이므로 중간 점수
        
        for dr, dc in directions:
            count = 1
            # 양쪽이 막혔는지 확인 (0: 안 막힘, 1: 막힘/벽)
            blocked_sides = 0
            
            # 정방향 탐색
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            if not (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0):
                blocked_sides += 1

            # 역방향 탐색
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            if not (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0):
                blocked_sides += 1
            
            # (1) 열린 4 (양쪽 뚫림 or 한쪽 뚫림) -> 다음 턴에 5가 될 수 있음
            if count == 4:
                if blocked_sides < 2: # 완전히 막힌 게 아니면
                    return 0.5 # 아주 큰 보상! (0.5승)

            # (2) 열린 3 (양쪽 다 뚫려야만 진짜 3)
            if count == 3:
                if blocked_sides == 0: # 양쪽 다 뚫려 있어야 함
                    return 0.2 # 기분 좋은 보상

        return 0.0

    def step(self, action):
        row = action // self.board_size
        col = action % self.board_size
        
        if self.board[row, col] != 0:
            return self.get_state(), -1, True # 잘못된 수

        self.board[row, col] = self.current_player

        # 1. 승리 체크
        if self.check_win(row, col, self.current_player):
            self.winner = self.current_player
            return self.get_state(), 1.0, True
        
        # 2. 무승부 체크
        if np.all(self.board != 0):
            return self.get_state(), 0, True
        
        # (!!!) 3. 중간 보상 체크 (승패가 안 났을 때)
        # 공격적인 수를 두면 보너스 점수 부여
        bonus = self.check_attack_pattern(row, col, self.current_player)
        
        self.current_player *= -1
        
        # 게임은 안 끝났지만(False), 보너스 점수를 반환
        return self.get_state(), bonus, False

    def copy(self):
        new_env = OmokEnv(self.board_size)
        new_env.board = np.copy(self.board)
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        return new_env
