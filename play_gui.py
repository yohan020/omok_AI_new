# 파일명: play_gui.py

import tkinter as tk
from tkinter import messagebox, simpledialog
import torch
import numpy as np
import threading
import random
import time

# 기존 모듈 임포트
from model import ResNetActorCritic
from environment import OmokEnv
from mcts import run_mcts

# --- 설정 ---
MODEL_PATH = 'models_pure_resnet/resnet_omok_model_cycle_490.pth' # (!!!) 모델 경로 확인
BOARD_SIZE = 10
MCTS_SIMULATIONS_PLAY = 1000 # AI 생각 깊이
CELL_SIZE = 50  # 격자 한 칸 크기 (픽셀)
PADDING = 30    # 여백

# --- 하이브리드 규칙 헬퍼 함수들 (AI 로직) ---
def check_winning_move(env, player):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if env.board[r, c] == 0:
                env.board[r, c] = player
                if env.check_win(r, c, player):
                    env.board[r, c] = 0
                    return r * BOARD_SIZE + c
                env.board[r, c] = 0
    return None

def get_moves_that_make_pattern(env, player, target_count, open_ends_required=2):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if env.board[r, c] == 0:
                env.board[r, c] = player
                for dr, dc in directions:
                    count = 1
                    blocked = 0
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == player:
                        count += 1; nr += dr; nc += dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == 0): blocked += 1
                    
                    nr, nc = r - dr, c - dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == player:
                        count += 1; nr -= dr; nc -= dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == 0): blocked += 1
                    
                    if count >= target_count and (2 - blocked) >= open_ends_required:
                        candidates.append(r * BOARD_SIZE + c)
                        break
                env.board[r, c] = 0
    return candidates

class OmokGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AlphaZero Omok (ResNet + Hybrid) - {BOARD_SIZE}x{BOARD_SIZE}")
        
        # 모델 및 환경 로드
        self.device = self.get_device()
        self.model = ResNetActorCritic(board_size=BOARD_SIZE).to(self.device)
        self.load_model()
        self.env = OmokEnv(board_size=BOARD_SIZE)
        
        # 게임 상태 변수
        self.human_color = 1 # 1: 흑, -1: 백
        self.game_over = False
        self.last_move = None # 마지막 착수 위치 (표시용)
        
        # 캔버스 생성
        canvas_width = BOARD_SIZE * CELL_SIZE + PADDING * 2
        canvas_height = BOARD_SIZE * CELL_SIZE + PADDING * 2
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="#E3C588") # 바둑판 색
        self.canvas.pack()
        
        # 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.handle_click)
        
        # 하단 상태바
        self.status_label = tk.Label(root, text="준비 중...", font=("Arial", 14))
        self.status_label.pack(pady=10)
        
        # 게임 시작
        self.start_game()

    def get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        elif torch.backends.mps.is_available(): return torch.device("mps")
        else: return torch.device("cpu")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            print(f"모델 로드 완료: {MODEL_PATH}")
        except FileNotFoundError:
            messagebox.showerror("오류", f"모델 파일을 찾을 수 없습니다:\n{MODEL_PATH}")
            self.root.destroy()

    def draw_board(self):
        self.canvas.delete("all")
        # 격자 그리기
        for i in range(BOARD_SIZE):
            # 가로줄
            self.canvas.create_line(PADDING, PADDING + i * CELL_SIZE,
                                    PADDING + (BOARD_SIZE - 1) * CELL_SIZE, PADDING + i * CELL_SIZE)
            # 세로줄
            self.canvas.create_line(PADDING + i * CELL_SIZE, PADDING,
                                    PADDING + i * CELL_SIZE, PADDING + (BOARD_SIZE - 1) * CELL_SIZE)
            
            # 좌표 텍스트 (옵션)
            self.canvas.create_text(PADDING - 15, PADDING + i * CELL_SIZE, text=str(i), fill="gray")
            self.canvas.create_text(PADDING + i * CELL_SIZE, PADDING - 15, text=str(i), fill="gray")

    def draw_stones(self):
        # 기존 돌은 유지하고 새로 그려진 것만 그릴 수도 있지만, 간단히 전체 다시 그리기
        # (최적화를 위해선 델타만 그리는 게 좋지만 10x10이라 괜찮음)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.env.board[r, c] != 0:
                    x = PADDING + c * CELL_SIZE
                    y = PADDING + r * CELL_SIZE
                    color = "black" if self.env.board[r, c] == 1 else "white"
                    # 돌 그리기
                    self.canvas.create_oval(x - 18, y - 18, x + 18, y + 18, fill=color, outline=color)
                    
                    # 마지막 수 표시 (빨간 점)
                    if self.last_move == (r, c):
                        self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill="red", outline="red")

    def start_game(self):
        self.env.reset()
        self.game_over = False
        self.last_move = None
        self.draw_board()
        
        # 선공/후공 선택
        choice = messagebox.askyesno("선공/후공 선택", "흑돌(선공)을 잡으시겠습니까?\n(No를 누르면 백돌/후공이 됩니다)")
        
        if choice: # 흑돌 (사람 선공)
            self.human_color = 1
            self.status_label.config(text="당신의 차례입니다 (흑돌)")
        else: # 백돌 (AI 선공)
            self.human_color = -1
            self.status_label.config(text="AI가 생각 중입니다...")
            self.root.after(500, self.run_ai_thread) # 약간의 딜레이 후 AI 착수

    def handle_click(self, event):
        if self.game_over:
            if messagebox.askyesno("게임 종료", "다시 하시겠습니까?"):
                self.start_game()
            return

        if self.env.current_player != self.human_color:
            return # AI 턴에는 클릭 무시

        # 클릭 좌표를 보드 좌표로 변환 (반올림하여 가장 가까운 교차점 찾기)
        col = round((event.x - PADDING) / CELL_SIZE)
        row = round((event.y - PADDING) / CELL_SIZE)

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            action = row * BOARD_SIZE + col
            if self.env.get_valid_moves()[action]:
                # 사람 착수
                self.process_move(action)
                
                if not self.game_over:
                    # AI 턴 시작
                    self.status_label.config(text="AI가 생각 중입니다...", fg="blue")
                    # GUI 멈춤 방지를 위해 스레드로 실행
                    threading.Thread(target=self.run_ai_thread, daemon=True).start()
            else:
                print("이미 돌이 있는 자리입니다.")

    def process_move(self, action):
        row, col = divmod(action, BOARD_SIZE)
        _, reward, done = self.env.step(action)
        self.last_move = (row, col)
        self.draw_stones()
        
        if done:
            self.game_over = True
            if reward == 1.0:
                winner = -self.env.current_player # 방금 둔 사람이 승자
                if winner == self.human_color:
                    self.status_label.config(text="승리! 축하합니다!", fg="green")
                    messagebox.showinfo("결과", "당신이 이겼습니다!")
                else:
                    self.status_label.config(text="패배... AI가 이겼습니다.", fg="red")
                    messagebox.showinfo("결과", "AI가 이겼습니다.")
            elif reward == -1.0:
                self.status_label.config(text="AI 착수 오류 (승리)", fg="green")
            else:
                self.status_label.config(text="무승부", fg="black")

    def run_ai_thread(self):
        """ AI 로직 (하이브리드) 실행 - 백그라운드 스레드 """
        action = self.get_hybrid_ai_move()
        
        # UI 업데이트는 메인 스레드에서 해야 함
        self.root.after(0, lambda: self.process_move(action))
        if not self.game_over:
            self.root.after(0, lambda: self.status_label.config(text="당신의 차례입니다", fg="black"))

    def get_hybrid_ai_move(self):
        """ play_mcts.py의 하이브리드 로직과 동일 """
        env = self.env # 현재 환경 참조
        ai_player = env.current_player
        opponent = -ai_player
        
        # 1. 킬각 (5목)
        win_move = check_winning_move(env, ai_player)
        if win_move is not None: return win_move
        
        # 2. 절대 방어 (상대 5목)
        block_win = check_winning_move(env, opponent)
        if block_win is not None: return block_win
        
        # 3. 필승 공격 (내 열린 4)
        my_open_4 = get_moves_that_make_pattern(env, ai_player, 4, 2)
        if my_open_4: return random.choice(my_open_4)
        
        # 4. 4목 방어
        opp_4 = get_moves_that_make_pattern(env, opponent, 4, 1)
        if opp_4: return opp_4[0]
        
        # 5. 사전 방어 (상대 열린 3)
        opp_open_3 = get_moves_that_make_pattern(env, opponent, 3, 2)
        if opp_open_3: return opp_open_3[0]
        
        # 6. 공격 전개 (내 열린 3)
        my_open_3 = get_moves_that_make_pattern(env, ai_player, 3, 2)
        if my_open_3: return random.choice(my_open_3)
        
        # 7. MCTS
        action, _ = run_mcts(env, self.model, self.device,
                             num_simulations=MCTS_SIMULATIONS_PLAY, c_puct=1.0)
        return action

if __name__ == "__main__":
    root = tk.Tk()
    # 창 크기 조절 불가하게 설정 (레이아웃 깨짐 방지)
    root.resizable(False, False)
    app = OmokGUI(root)
    root.mainloop()
