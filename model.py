# 파일명: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """ ResNet의 기본 블록: 입력을 출력에 더해줌으로써 깊은 층도 학습 가능하게 함 """
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual # 핵심: 스킵 연결 (Skip Connection)
        x = F.relu(x)
        return x

class ResNetActorCritic(nn.Module):
    def __init__(self, board_size=10, num_res_blocks=4, num_channels=64):
        super(ResNetActorCritic, self).__init__()
        self.board_size = board_size
        
        # 1. 초기 컨볼루션
        self.start_conv = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.start_bn = nn.BatchNorm2d(num_channels)
        
        # 2. 레지듀얼 타워 (몸통)
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 3. 정책 헤드 (Policy Head) - 어디에 둘지 결정
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1) # 채널을 2로 줄임
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 4. 가치 헤드 (Value Head) - 승률 예측
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1) # 채널을 1로 줄임
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # 입력: (N, 2, H, W)
        x = F.relu(self.start_bn(self.start_conv(x)))
        
        # 몸통 통과
        for block in self.res_blocks:
            x = block(x)
            
        # 정책 헤드
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p) # Logits (Softmax 전)
        
        # 가치 헤드
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # -1 ~ 1 사이 값
        
        return p, v
