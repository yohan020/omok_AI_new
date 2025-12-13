# 파일명: play_gui.py

import tkinter as tk
from tkinter import messagebox, simpledialog
import torch
import numpy as np
import threading
import random
import time

# (!!!) ResNet 모델 임포트
from model import ResNetActorCritic
from environment import OmokEnv
from mcts import run_mcts

# --- 설정 ---
MODEL_PATH = 'models_8x8_reward/resnet_omok_model_cycle_330.pth' # (!!!) 모델 경로 확인
BOARD_SIZE = 8
MCTS_SIMULATIONS_PLAY = 4000 # AI 생각 깊이 (순수 AI이므로 높게 설정 권장)
CELL_SIZE = 50  # 격자 한 칸 크기 (픽셀)
PADDING = 30    # 여백

# --- (!!!) 하이브리드 규칙 헬퍼 함수들 (모두 제거됨) ---
# check_winning_move 함수 제거됨
# get_moves_that_make_pattern 함수 제거됨

class OmokGUI:
    def __init__(self, root):
        self.root = root
        # (!!!) 제목에서 하이브리드 제거
        self.root.title(f"AlphaZero Omok (Pure ResNet) - {BOARD_SIZE}x{BOARD_SIZE}")
        
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
            
            # 좌표 텍스트
            self.canvas.create_text(PADDING - 15, PADDING + i * CELL_SIZE, text=str(i), fill="gray")
            self.canvas.create_text(PADDING + i * CELL_SIZE, PADDING - 15, text=str(i), fill="gray")

    def draw_stones(self):
        # 전체 돌 다시 그리기
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

        # 클릭 좌표를 보드 좌표로 변환
        col = round((event.x - PADDING) / CELL_SIZE)
        row = round((event.y - PADDING) / CELL_SIZE)

        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            action = row * BOARD_SIZE + col
            if self.env.get_valid_moves()[action]:
                # 사람 착수
                self.process_move(action)
                
                if not self.game_over:
                    # AI 턴 시작
                    self.status_label.config(text="AI가 MCTS 수읽기 중...", fg="blue")
                    # GUI 멈춤 방지를 위해 스레드로 실행
                    threading.Thread(target=self.run_ai_thread, daemon=True).start()
            else:
                print("이미 돌이 있는 자리입니다.")

    def process_move(self, action):
        row, col = divmod(action, BOARD_SIZE)
        
        # (!!!) 방금 둔 플레이어를 미리 저장 (env.step 이후에 턴이 바뀔 수 있으므로)
        current_mover = self.env.current_player
        
        _, reward, done = self.env.step(action)
        self.last_move = (row, col)
        self.draw_stones()
        
        if done:
            self.game_over = True
            
            # (!!!) 승패 판별 로직 수정
            # reward가 1.0이면 '방금 둔 사람(current_mover)'이 이긴 것
            if reward >= 1.0: # (중간 보상 때문에 1.0보다 클 수도 있음)
                winner = current_mover
                if winner == self.human_color:
                    self.status_label.config(text="승리! 축하합니다!", fg="green")
                    messagebox.showinfo("결과", "당신이 이겼습니다!")
                else:
                    self.status_label.config(text="패배... AI가 이겼습니다.", fg="red")
                    messagebox.showinfo("결과", "AI가 이겼습니다.")
            
            elif reward == -1.0: # 착수 오류 등
                self.status_label.config(text="오류 발생", fg="red")
            else:
                self.status_label.config(text="무승부", fg="black")

    def run_ai_thread(self):
        """ AI 로직 (순수 MCTS) 실행 - 백그라운드 스레드 """
        
        start_time = time.time()
        
        # (!!!) 수정된 부분: add_noise=False 추가
        # 대결 모드에서는 확률적 선택(탐험)을 끄고, 가장 좋은 수(Argmax)만 선택하게 함
        action, pi_target = run_mcts(self.env, self.model, self.device,
                                     num_simulations=MCTS_SIMULATIONS_PLAY,
                                     c_puct=1.0,
                                     add_noise=False)
        
        end_time = time.time()
        
        # (이하 동일)
        if action != -1:
            row, col = divmod(action, BOARD_SIZE)
            conf = pi_target[action] * 100 if pi_target is not None else 0
            print(f"AI 착수: ({row}, {col}) | 확신: {conf:.1f}% | 소요시간: {end_time - start_time:.1f}초")
        else:
            print("AI가 기권했습니다.")
            
        if action != -1:
            self.root.after(0, lambda: self.process_move(action))
            if not self.game_over:
                self.root.after(0, lambda: self.status_label.config(text="당신의 차례입니다", fg="black"))
        else:
            self.root.after(0, lambda: self.status_label.config(text="AI 기권 (당신 승리)", fg="green"))

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = OmokGUI(root)
    root.mainloop()
