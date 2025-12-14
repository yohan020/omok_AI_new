# 파일명: play_ai_vs_ai.py

import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import threading
import time

# 모델 및 환경 임포트
from model import ResNetActorCritic
from environment import OmokEnv
from mcts import run_mcts

# --- 설정 (대결시킬 모델 2개 선택) ---
BOARD_SIZE = 8
MCTS_SIMULATIONS = 4000  # AI 수읽기 횟수 (높을수록 진지함)
DELAY_BETWEEN_MOVES = 500 # 착수 간 딜레이 (ms) - 너무 빠르면 안 보이니까

# 흑돌(Black, 선공) 모델
MODEL_A_PATH = 'models_8x8_reward/resnet_omok_model_cycle_580.pth'

# 백돌(White, 후공) 모델 - (예: 과거 버전 or 동일 버전)
MODEL_B_PATH = 'models_8x8_reward/resnet_omok_model_cycle_580.pth'

# -------------------------------------

class AIvsAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"AI vs AI Match ({BOARD_SIZE}x{BOARD_SIZE})")
        
        # 설정
        self.cell_size = 50
        self.padding = 30
        
        # 디바이스 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- 두 개의 모델 로드 ---
        self.model_A = self.load_model(MODEL_A_PATH, "흑돌(A)")
        self.model_B = self.load_model(MODEL_B_PATH, "백돌(B)")
        
        # 환경 초기화
        self.env = OmokEnv(board_size=BOARD_SIZE)
        self.game_over = False
        self.last_move = None
        self.is_thinking = False # 중복 실행 방지

        # GUI 구성
        canvas_width = BOARD_SIZE * self.cell_size + self.padding * 2
        canvas_height = BOARD_SIZE * self.cell_size + self.padding * 2 + 50
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="#E3C588")
        self.canvas.pack()
        
        # 정보 표시 레이블
        self.info_label = tk.Label(root, text="준비...", font=("Arial", 14, "bold"), bg="#E3C588")
        self.info_label.place(x=self.padding, y=canvas_height-40)

        # 게임 시작
        self.start_game()

    def load_model(self, path, name):
        model = ResNetActorCritic(board_size=BOARD_SIZE).to(self.device)
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            print(f"[{name}] 모델 로드 성공: {path}")
        except FileNotFoundError:
            print(f"[{name}] 오류: 모델 파일을 찾을 수 없습니다 -> {path}")
            messagebox.showerror("오류", f"{name} 모델을 찾을 수 없습니다.")
            self.root.destroy()
        return model

    def draw_board(self):
        self.canvas.delete("all")
        # 격자 그리기
        for i in range(BOARD_SIZE):
            # 가로
            self.canvas.create_line(self.padding, self.padding + i * self.cell_size,
                                    self.padding + (BOARD_SIZE - 1) * self.cell_size, self.padding + i * self.cell_size)
            # 세로
            self.canvas.create_line(self.padding + i * self.cell_size, self.padding,
                                    self.padding + i * self.cell_size, self.padding + (BOARD_SIZE - 1) * self.cell_size)
            
            # 좌표
            self.canvas.create_text(self.padding - 15, self.padding + i * self.cell_size, text=str(i), fill="gray")
            self.canvas.create_text(self.padding + i * self.cell_size, self.padding - 15, text=str(i), fill="gray")

    def draw_stones(self):
        # 돌 그리기
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.env.board[r, c] != 0:
                    x = self.padding + c * self.cell_size
                    y = self.padding + r * self.cell_size
                    color = "black" if self.env.board[r, c] == 1 else "white"
                    outline = "black"
                    self.canvas.create_oval(x - 20, y - 20, x + 20, y + 20, fill=color, outline=outline)
                    
                    # 마지막 수 표시
                    if self.last_move == (r, c):
                        self.canvas.create_rectangle(x-5, y-5, x+5, y+5, fill="red", outline="red")

    def start_game(self):
        self.env.reset()
        self.game_over = False
        self.last_move = None
        self.draw_board()
        self.info_label.config(text="대국 시작! 흑돌(A) 차례")
        
        # 1초 뒤 첫 수 시작
        self.root.after(1000, self.play_next_turn)

    def play_next_turn(self):
        """ 다음 턴 진행 (스레드 시작) """
        if self.game_over: return
        
        current_player_name = "흑돌(A)" if self.env.current_player == 1 else "백돌(B)"
        self.info_label.config(text=f"{current_player_name} 생각 중...", fg="blue")
        
        # 백그라운드 스레드에서 MCTS 실행
        threading.Thread(target=self.run_ai_logic, daemon=True).start()

    def run_ai_logic(self):
        """ 현재 턴의 AI가 MCTS로 수를 결정 """
        
        # 현재 턴에 맞는 모델 선택
        if self.env.current_player == 1:
            current_model = self.model_A
            p_name = "Black(A)"
        else:
            current_model = self.model_B
            p_name = "White(B)"

        start_t = time.time()
        
        # (!!!) 대결 모드이므로 add_noise=False 필수!
        action, pi = run_mcts(self.env, current_model, self.device,
                              num_simulations=MCTS_SIMULATIONS,
                              c_puct=1.0,
                              add_noise=False)
        
        end_t = time.time()
        
        # 결과 처리 (메인 스레드로 전달)
        self.root.after(0, lambda: self.apply_move(action, pi, p_name, end_t - start_t))

    def apply_move(self, action, pi, p_name, duration):
        """ 결정된 수를 보드에 반영하고 승패 판정 """
        if self.game_over: return

        if action == -1:
            self.game_over = True
            self.info_label.config(text=f"{p_name} 기권! 게임 종료", fg="red")
            return

        row, col = divmod(action, BOARD_SIZE)
        conf = pi[action] * 100 if pi is not None else 0
        
        print(f"🤖 {p_name} 착수: ({row}, {col}) | 확신: {conf:.1f}% | 시간: {duration:.1f}초")

        # 환경 업데이트
        _, reward, done = self.env.step(action)
        self.last_move = (row, col)
        self.draw_stones()

        if done:
            self.game_over = True
            
            # (!!!) 수정된 승패 판정 로직
            # p_name: 방금 착수한 플레이어의 이름 (Black(A) or White(B))
            # reward가 1.0 이상이면 방금 둔 플레이어가 이긴 것임
            
            if reward >= 1.0:
                winner_text = f"🎉 {p_name} 승리! 🎉"
                fg_color = "red"
            elif reward == -1.0: # 착수 오류 등
                winner_text = f"{p_name} 반칙패 (오류)"
                fg_color = "black"
            else:
                winner_text = "무승부"
                fg_color = "black"
            
            self.info_label.config(text=winner_text, fg=fg_color)
            messagebox.showinfo("결과", winner_text)
        else:
            # 다음 턴 예약 (사람이 볼 수 있게 딜레이)
            self.root.after(DELAY_BETWEEN_MOVES, self.play_next_turn)

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = AIvsAIGUI(root)
    root.mainloop()
