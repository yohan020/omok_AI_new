# 파일명: mcts.py

import numpy as np
import math
import torch
import torch.nn.functional as F

class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.P = prior_p

    def get_Q(self):
        return self.W / self.N if self.N > 0 else 0

    def get_U(self, c_puct):
        parent_N_sqrt = math.sqrt(self.parent.N) if self.parent else 1
        return c_puct * self.P * (parent_N_sqrt / (1 + self.N))

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.get_Q() + child.get_U(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = Node(parent=self, prior_p=prob)

    def backpropagate(self, value):
        self.N += 1
        self.W += value
        if self.parent:
            self.parent.backpropagate(-value)

# (!!!) 매개변수에 add_noise가 추가되었습니다.
def run_mcts(env, model, device, num_simulations=100, c_puct=1.0, add_noise=False):
    
    root_node = Node(parent=None, prior_p=1.0)
    
    # --- 1. 루트 노드 먼저 확장 (노이즈 추가를 위해) ---
    current_state_tensor = torch.tensor(env.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
    valid_moves_mask = torch.tensor(env.get_valid_moves(), dtype=torch.bool).to(device)

    with torch.no_grad():
        policy_logits, _ = model(current_state_tensor)
    
    policy_logits[0, ~valid_moves_mask] = -float('inf')
    action_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
    root_node.expand(action_probs)

    # (!!!) 디리클레 노이즈 추가 (훈련 시 창의성 부여)
    if add_noise:
        epsilon = 0.25  # 노이즈 비율 (25%는 랜덤성)
        alpha = 0.03    # 노이즈의 분산 (오목/바둑은 보통 0.03)
        noises = np.random.dirichlet([alpha] * len(root_node.children))
        
        for i, (action, child) in enumerate(root_node.children.items()):
            child.P = (1 - epsilon) * child.P + epsilon * noises[i]
    # ------------------------------------------------

    for _ in range(num_simulations):
        node = root_node
        sim_env = env.copy()
        path = [node]
        
        done = False
        reward = 0.0

        # Select
        while node.children:
            action, node = node.select_child(c_puct)
            _, reward, done = sim_env.step(action)
            path.append(node)
            if done: break
        
        # Expand & Evaluate
        if not done:
            # 이미 루트에서 확장을 했으므로, 여기선 리프 노드일 때만 모델 실행
            current_state_tensor = torch.tensor(sim_env.get_state(), dtype=torch.float32).unsqueeze(0).to(device)
            valid_moves_mask = torch.tensor(sim_env.get_valid_moves(), dtype=torch.bool).to(device)

            with torch.no_grad():
                policy_logits, value_tensor = model(current_state_tensor)
            
            value = value_tensor.item()
            
            policy_logits[0, ~valid_moves_mask] = -float('inf')
            action_probs = F.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            node.expand(action_probs)
        else:
            value = reward

        # Backpropagate
        for node_in_path in reversed(path):
            node_in_path.N += 1
            node_in_path.W += value
            value = -value

    visit_counts = []
    actions = []
    for action, child in root_node.children.items():
        actions.append(action)
        visit_counts.append(child.N)

    if not actions: return -1, None

    visit_counts = np.array(visit_counts)
    actions = np.array(actions)
    
    # (!!!) 훈련(add_noise=True)일 때는 확률적으로 선택 (Temperature)
    # 대결(add_noise=False)일 때는 가장 많이 방문한 수 선택 (Argmax)
    if add_noise:
        # 방문 횟수에 비례하여 확률적으로 선택 (다양한 오프닝 학습)
        pi_target_probs = visit_counts / np.sum(visit_counts)
        best_action = np.random.choice(actions, p=pi_target_probs)
    else:
        best_action = actions[np.argmax(visit_counts)]
        pi_target_probs = visit_counts / np.sum(visit_counts)

    pi_target = np.zeros(env.board_size * env.board_size)
    pi_target[actions] = pi_target_probs
    
    return best_action, pi_target
