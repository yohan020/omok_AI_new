# 파일명: play_mcts.py

import torch
import numpy as np
import time
import random

# (!!!) 중요: 훈련 중인 ResNet 모델 사용
from model import ResNetActorCritic
from environment import OmokEnv
from mcts import run_mcts

# (!!!) 방금 확인한 최신 모델 경로로 수정하세요
MODEL_PATH = 'models_pure_resnet/resnet_omok_model_cycle_185.pth'
BOARD_SIZE = 10
MCTS_SIMULATIONS_PLAY = 2000 # 생각할 시간

def print_board(board):
    print("   " + " ".join([f"{i:2}" for i in range(BOARD_SIZE)]))
    print("  " + "-" * (BOARD_SIZE * 3 - 1))
    for r in range(BOARD_SIZE):
        row_str = f"{r:2}|"
        for c in range(BOARD_SIZE):
            if board[r, c] == 1: row_str += " B "
            elif board[r, c] == -1: row_str += " W "
            else: row_str += " . "
        print(row_str)

def get_human_move(env, player_color):
    while True:
        try:
            move_str = input(f"당신의 차례입니다 ({player_color}). (row, col) 입력 (0~{BOARD_SIZE-1}): ")
            row, col = map(int, move_str.split(','))
            action = row * BOARD_SIZE + col
            
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                print("보드 범위를 벗어났습니다.")
            elif not env.get_valid_moves()[action]:
                print("이미 돌이 있는 곳입니다.")
            else:
                return action
        except ValueError:
            print("잘못된 형식입니다. 예: 5, 5")

# --- [규칙] 헬퍼 함수들 ---
def check_winning_move(env, player):
    """ 1. 킬각 확인 """
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
    """ 2. 특정 패턴(열린 3, 열린 4)을 만드는 자리 찾기 """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    candidates = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if env.board[r, c] == 0:
                env.board[r, c] = player
                for dr, dc in directions:
                    count = 1
                    blocked = 0
                    # 정방향
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == player:
                        count += 1; nr += dr; nc += dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == 0): blocked += 1
                    # 역방향
                    nr, nc = r - dr, c - dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == player:
                        count += 1; nr -= dr; nc -= dc
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and env.board[nr, nc] == 0): blocked += 1
                    
                    if count >= target_count and (2 - blocked) >= open_ends_required:
                        candidates.append(r * BOARD_SIZE + c)
                        break
                env.board[r, c] = 0
    return candidates

def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    # (!!!) 훈련 중인 ResNet 모델 로드
    model = ResNetActorCritic(board_size=BOARD_SIZE).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"모델 로드 성공: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 -> {MODEL_PATH}")
        return
        
    model.eval()
    env = OmokEnv(board_size=BOARD_SIZE)
    state = env.reset()
    done = False
    
    human_player = 0
    human_color_str = ""
    ai_color_str = ""
    while human_player == 0:
        choice = input("흑돌(B, 선공) 또는 백돌(W, 후공)을 선택하세요: ").upper()
        if choice == 'B': human_player = 1; human_color_str = "흑돌"; ai_color_str = "백돌"
        elif choice == 'W': human_player = -1; human_color_str = "백돌"; ai_color_str = "흑돌"
    
    print(f"\n당신은 {human_color_str}, AI는 {ai_color_str}입니다.")
    
    if human_player == -1:
        print_board(env.board)
        print(f"\n🤖 AI({ai_color_str})가 첫 수를 둡니다...")
        # 첫 수는 순수 모델의 판단(MCTS)에 맡김
        action, _ = run_mcts(env, model, device, num_simulations=MCTS_SIMULATIONS_PLAY, c_puct=1.0)
        row, col = divmod(action, BOARD_SIZE)
        print(f"🤖 AI 착수: ({row}, {col})")
        state, _, _ = env.step(action)

    while not done:
        print_board(env.board)
        
        if env.current_player == human_player:
            action = get_human_move(env, human_color_str)
        else:
            print(f"\n🤖 AI({ai_color_str})가 생각 중입니다...")
            ai_player = env.current_player
            opponent = -ai_player
            action = -1
            
            # --- [규칙 1] 킬각 (5목) ---
            win_move = check_winning_move(env, ai_player)
            if win_move is not None:
                print("⚡ AI: 체크메이트! (승리)")
                action = win_move
            
            # --- [규칙 2] 절대 방어 (상대 5목 저지) ---
            if action == -1:
                block_win = check_winning_move(env, opponent)
                if block_win is not None:
                    print("🛡️ AI: 5목 방어!")
                    action = block_win

            # --- [규칙 3] 필승 공격 (내 열린 4 만들기) ---
            if action == -1:
                my_open_4 = get_moves_that_make_pattern(env, ai_player, 4, 2)
                if my_open_4:
                    print("⚔️ AI: 필승 공격 (열린 4)")
                    action = random.choice(my_open_4)

            # --- [규칙 4] 4목 방어 ---
            if action == -1:
                # 상대가 두면 4개가 되는데, 한쪽이라도 뚫려있으면(Open>=1) 막아야 함
                opp_4 = get_moves_that_make_pattern(env, opponent, 4, 1)
                if opp_4:
                    print("🛡️ AI: 4목 방어!")
                    action = opp_4[0]

            # --- [규칙 5] 사전 방어 (상대 열린 3) ---
            if action == -1:
                opp_open_3 = get_moves_that_make_pattern(env, opponent, 3, 2)
                if opp_open_3:
                    print("🛡️ AI: 3목 견제")
                    action = opp_open_3[0]

            # --- [규칙 6] 내 열린 3 만들기 (공격) ---
            if action == -1:
                my_open_3 = get_moves_that_make_pattern(env, ai_player, 3, 2)
                if my_open_3:
                    print("⚔️ AI: 공격 전개 (열린 3)")
                    action = random.choice(my_open_3)

            # --- [본능] 훈련된 모델의 MCTS 수읽기 ---
            if action == -1:
                print(f"(MCTS 수읽기 {MCTS_SIMULATIONS_PLAY}회 진행 중...)")
                start_time = time.time()
                # (!!!) 여기서 훈련된 ResNet이 "가장 유리한 자리"를 찾아냅니다.
                action, pi_target = run_mcts(env, model, device,
                                             num_simulations=MCTS_SIMULATIONS_PLAY,
                                             c_puct=1.0)
                end_time = time.time()
                conf = pi_target[action] * 100 if pi_target is not None and action != -1 else 0
                print(f"   -> MCTS 완료 ({end_time - start_time:.1f}초). 확신: {conf:.1f}%")

            if action == -1:
                print("AI 기권")
                break

            row, col = divmod(action, BOARD_SIZE)
            print(f"🤖 AI 착수: ({row}, {col})")

        state, reward, done = env.step(action)
        
        if done:
            print_board(env.board)
            if reward == 1.0:
                winner = env.current_player * -1
                if winner == human_player: print(f"\n🎉 당신({human_color_str}) 승리!")
                else: print(f"\n🤖 AI({ai_color_str}) 승리!")
            else:
                print("\n무승부/오류!")

if __name__ == '__main__':
    main()
