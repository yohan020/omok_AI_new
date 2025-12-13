# 파일명: train_mcts.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
from tqdm import tqdm
import os
import random

from model import ResNetActorCritic
from environment import OmokEnv
from mcts import run_mcts

# --- 1. 훈련 설정 (8x8 버전) ---
BOARD_SIZE = 8              # (!!!) 8x8 보드
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 30000
EPISODES_PER_CYCLE = 20     # 한 사이클당 20판 대국
TRAIN_EPOCHS_PER_CYCLE = 10 # 한 사이클당 10회 학습
MCTS_SIMULATIONS = 400      # 8x8에서는 충분한 깊이
C_PUCT = 1.0
MODEL_SAVE_DIR = 'models_8x8_reward' # 저장 경로

# --- 2. 리셋 및 이어하기 설정 ---
RESUME_FROM_CYCLE = 0       # (!!!) 0부터 새로 시작 권장
FINAL_CYCLE_GOAL = 1000     # 목표 사이클

INITIAL_LEARNING_RATE = 0.001 # 초기 학습률
NEW_SCHEDULER_STEP = 100      # 100 사이클마다 학습률 감소
# -----------------------------

def get_symmetries(state, pi):
    """ 데이터 증강: 회전 및 대칭으로 1판을 8판처럼 만듦 """
    aug_data = []
    pi_board = np.reshape(pi, (BOARD_SIZE, BOARD_SIZE))
    for i in range(4):
        state_rot = np.rot90(state, k=i, axes=(1, 2))
        pi_rot = np.rot90(pi_board, k=i)
        aug_data.append((np.ascontiguousarray(state_rot), np.ascontiguousarray(pi_rot.flatten())))
        state_flip = np.flip(state_rot, axis=2)
        pi_flip = np.fliplr(pi_rot)
        aug_data.append((np.ascontiguousarray(state_flip), np.ascontiguousarray(pi_flip.flatten())))
    return aug_data

def self_play(model, device):
    """ 자가 대국: 데이터 수집 (가상 울타리 + 중간 보상 적용) """
    replay_data = []
    env = OmokEnv(board_size=BOARD_SIZE)
    state = env.reset()
    game_history = []
    
    move_count = 0
    
    # (!!!) 가상 울타리 설정: 초반 6수는 중앙 4x4 (인덱스 2~5) 강제
    RESTRICT_MOVES_UNTIL = 6
    MIN_IDX, MAX_IDX = 2, 6

    while True:
        # MCTS 실행 (노이즈 켜기: 탐험 유도)
        best_action, pi_target = run_mcts(env, model, device,
                                          num_simulations=MCTS_SIMULATIONS,
                                          c_puct=C_PUCT,
                                          add_noise=True)

        if best_action == -1: break
        
        # (!!!) 가상 울타리 강제 로직
        if move_count < RESTRICT_MOVES_UNTIL:
            row, col = divmod(best_action, BOARD_SIZE)
            # AI가 울타리 밖(구석)으로 나가려 하면?
            if not (MIN_IDX <= row < MAX_IDX and MIN_IDX <= col < MAX_IDX):
                # 강제로 중앙 빈칸 중 하나를 랜덤 선택 (교정)
                center_candidates = []
                for r in range(MIN_IDX, MAX_IDX):
                    for c in range(MIN_IDX, MAX_IDX):
                        if env.board[r, c] == 0:
                            center_candidates.append(r * BOARD_SIZE + c)
                
                if center_candidates:
                    best_action = random.choice(center_candidates)
                    # 정책 타겟도 이 수가 100% 정답인 것처럼 수정
                    pi_target = np.zeros(BOARD_SIZE * BOARD_SIZE)
                    pi_target[best_action] = 1.0
        
        # (!!!) 중간 보상 획득
        # step 함수가 반환하는 reward에는 승패(+1/-1) 뿐만 아니라
        # 공격 성공 보너스(+0.2, +0.5)도 포함됨 (environment.py 수정 필수)
        next_state, immediate_reward, done = env.step(best_action)
        
        # 역사 저장: (상태, MCTS확률, "이번 수의 보상")
        game_history.append([env.get_state(), pi_target, immediate_reward])
        
        state = next_state
        move_count += 1
        
        if done:
            # 역전파: 게임 끝에서부터 거슬러 올라가며 가치(Value) 계산
            running_value = 0.0
            
            for i in range(len(game_history) - 1, -1, -1):
                state_hist, pi_hist, reward_hist = game_history[i]
                
                if i == len(game_history) - 1:
                    # 마지막 수는 승패 보상 (+1.0 or -1.0 or 0)
                    running_value = reward_hist
                    if running_value > 1.0: running_value = 1.0 # 캡
                else:
                    # 중간 수는 "미래 가치(상대방 입장의 -Value)" + "즉각 보상(공격 점수)"
                    # V(s) = Reward + V(s') * (-1)
                    running_value = reward_hist - running_value
                
                # 데이터 증강 후 저장
                symmetries = get_symmetries(state_hist, pi_hist)
                for sym_state, sym_pi in symmetries:
                    # running_value가 신경망이 예측해야 할 목표값(z)이 됨
                    replay_data.append((sym_state, sym_pi, running_value))
            
            break
            
    return replay_data

def train_network(model, optimizer, replay_buffer, device):
    """ 신경망 학습 """
    sample_size = min(len(replay_buffer), BATCH_SIZE)
    if sample_size == 0: return 0.0, 0.0
        
    samples = random.sample(replay_buffer, sample_size)
    states, pis, zs = zip(*samples)
    
    state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    pi_target_batch = torch.tensor(np.array(pis), dtype=torch.float32).to(device)
    z_target_batch = torch.tensor(np.array(zs), dtype=torch.float32).unsqueeze(1).to(device)

    policy_logits, value_pred = model(state_batch)
    
    # Value Loss (MSE): 예측 가치와 실제 가치(보상 합계) 차이
    value_loss = F.mse_loss(value_pred, z_target_batch)
    
    # Policy Loss (Cross Entropy): 예측 확률과 MCTS 확률 차이
    policy_loss = -torch.sum(pi_target_batch * F.log_softmax(policy_logits, dim=-1), dim=-1).mean()
    
    total_loss = value_loss + policy_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return value_loss.item(), policy_loss.item()

def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        print(f"'{MODEL_SAVE_DIR}' 폴더를 생성했습니다.")

    model = ResNetActorCritic(board_size=BOARD_SIZE).to(device)
    
    start_cycle = 0
    current_lr = INITIAL_LEARNING_RATE

    # 이어하기 로직
    if RESUME_FROM_CYCLE > 0:
        MODEL_PATH = os.path.join(MODEL_SAVE_DIR, f'resnet_omok_model_cycle_{RESUME_FROM_CYCLE}.pth')
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"🔄 기존 훈련 모델 발견! 이어하기를 준비합니다: {MODEL_PATH}")
            start_cycle = RESUME_FROM_CYCLE
        except FileNotFoundError:
            print(f"모델({MODEL_PATH})이 없어 0부터 새로 시작합니다.")
    else:
        print("🚀 8x8 보드에서 훈련을 처음부터 시작합니다!")

    optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=NEW_SCHEDULER_STEP, gamma=0.1)
    
    print(f"   -> Cycle {start_cycle}부터 {FINAL_CYCLE_GOAL}까지 훈련합니다.")
    
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    for cycle in tqdm(range(start_cycle, FINAL_CYCLE_GOAL), desc="Training Progress"):
        
        # 자동 학습률 리셋 (0이 되면 초기화)
        current_lr = scheduler.get_last_lr()[0]
        if current_lr < 1e-8:
            print("\n" + "="*50)
            print(f"Cycle {cycle+1}: 학습률 리셋")
            optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=NEW_SCHEDULER_STEP, gamma=0.1)
            current_lr = scheduler.get_last_lr()[0]
            print(f"   -> 새 학습률: {current_lr:.8f}")
            print("="*50 + "\n")

        print(f"\n--- Cycle {cycle + 1}/{FINAL_CYCLE_GOAL} ---")
        
        # 1. 자가 대국 (데이터 수집)
        model.eval()
        pbar_self_play = tqdm(range(EPISODES_PER_CYCLE), desc="Self-Playing")
        for _ in pbar_self_play:
            new_data = self_play(model, device)
            replay_buffer.extend(new_data)
        
        # 2. 신경망 학습
        model.train()
        if len(replay_buffer) < BATCH_SIZE:
            print("데이터가 부족하여 훈련을 건너뜁니다.")
            scheduler.step()
            continue
            
        pbar_train = tqdm(range(TRAIN_EPOCHS_PER_CYCLE), desc="Training Network")
        total_v_loss = 0
        total_p_loss = 0
        for _ in pbar_train:
            v_loss, p_loss = train_network(model, optimizer, replay_buffer, device)
            total_v_loss += v_loss
            total_p_loss += p_loss
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        print(f"Avg Value Loss: {total_v_loss/TRAIN_EPOCHS_PER_CYCLE:.4f}, "
              f"Avg Policy Loss: {total_p_loss/TRAIN_EPOCHS_PER_CYCLE:.4f}, "
              f"LR: {current_lr:.8f}")

        # 모델 저장 (10 사이클마다)
        if (cycle + 1) % 10 == 0:
            save_path = os.path.join(MODEL_SAVE_DIR, f'resnet_omok_model_cycle_{cycle+1}.pth')
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"\nModel saved to {save_path}")

    # 최종 저장
    final_save_path = os.path.join(MODEL_SAVE_DIR, f'resnet_omok_model_{FINAL_CYCLE_GOAL}.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f"Final model saved to {final_save_path}")

if __name__ == '__main__':
    main()
