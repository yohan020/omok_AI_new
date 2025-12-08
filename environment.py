# 파일명: environment.py

import numpy as np

class OmokEnv:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1 # 1: 흑, -1: 백

    def reset(self):
        """ 보드를 리셋하고 초기 상태를 반환합니다. """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_valid_moves(self):
        """ 둘 수 있는 위치(빈 칸)를 1D 마스크로 반환 (True/False) """
        return (self.board == 0).flatten()

    def get_state(self):
        """ 모델 입력을 위한 2채널 상태 반환 (현재 플레이어, 상대방) """
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def check_win(self, r, c, player):
        """ (r, c)에 놓인 돌을 기준으로 5목을 확인합니다. """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # 가로, 세로, 대각선, 역대각선
        for dr, dc in directions:
            count = 1
            # 한 방향
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            # 반대 방향
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    # (!!!) MCTS 시뮬레이션을 위해 추가된 메서드
    def copy(self):
        """ MCTS 시뮬레이션을 위한 환경 복사본 생성 """
        new_env = OmokEnv(self.board_size)
        new_env.board = np.copy(self.board)
        new_env.current_player = self.current_player
        return new_env

    # (!!!) MCTS 호환을 위해 수정된 step 메서드
    def step(self, action):
        """
        MCTS 호환 step 함수.
        보상은 "방금 수를 둔 플레이어" 기준입니다.
        (승리: 1.0, 패배: -1.0, 무승부: 0.0, 진행: 0.0)
        """
        row, col = divmod(action, self.board_size)

        # 1. 유효하지 않은 수(이미 돌이 있음)
        if self.board[row, col] != 0:
            reward = -1.0 # 잘못된 수를 둔 현재 플레이어의 즉각 패배
            done = True
            # 턴을 넘기지 않아야 현재 플레이어가 패배자가 됨
            return self.get_state(), reward, done

        # 2. 돌을 놓음
        self.board[row, col] = self.current_player

        # 3. 방금 둔 수로 승리
        if self.check_win(row, col, self.current_player):
            reward = 1.0 # 현재 플레이어 승리
            done = True
        # 4. 무승부
        elif np.all(self.board != 0):
            reward = 0.0 # 무승부
            done = True
        # 5. 게임 계속
        else:
            reward = 0.0 # 진행 중
            done = False

        self.current_player *= -1 # 턴 넘김
        
        return self.get_state(), reward, done