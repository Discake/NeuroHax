"""
Контроллер для управления игроком с клавиатуры.

Team 1 (красный, левый):  W/S/A/D + Space (удар)
Team 2 (синий, правый):   стрелки Up/Down/Left/Right + Enter (удар)

Важно: для team 2 клавиши Left/Right меняются местами в action-векторе,
потому что SimpleModelNetAction применяет x_flip=-1 к dx.
Это означает что action[right]=1 даёт dx=+1, а затем target_vx = +1*(-1)*speed = -speed (влево на экране).
Поэтому Left arrow → action[right]=1, Right arrow → action[left]=1,
что после x_flip=-1 даёт правильное движение на экране.
"""

import pygame
import torch
import Constants


class HumanPlayerController:
    """
    Читает текущее состояние клавиатуры и возвращает action-тензор
    формата [up, down, left, right, kick] (бинарный, float32).
    """

    CONTROLS_TEAM1 = "Player 1 (YOU): W/S/A/D + SPACE (kick)"
    CONTROLS_TEAM2 = "Player 2 (YOU): Arrows + ENTER (kick)"

    def __init__(self, is_team_1: bool):
        self.is_team_1 = is_team_1

    def get_action(self) -> torch.Tensor:
        keys = pygame.key.get_pressed()

        if self.is_team_1:
            up    = 1.0 if keys[pygame.K_w] else 0.0
            down  = 1.0 if keys[pygame.K_s] else 0.0
            left  = 1.0 if keys[pygame.K_a] else 0.0
            right = 1.0 if keys[pygame.K_d] else 0.0
            kick  = 1.0 if keys[pygame.K_SPACE] else 0.0
        else:
            up   = 1.0 if keys[pygame.K_UP] else 0.0
            down = 1.0 if keys[pygame.K_DOWN] else 0.0
            # Swap left/right: RIGHT arrow должна двигать вправо на экране,
            # но SimpleModelNetAction умножает dx на x_flip=-1 для team2.
            # Поэтому нажатие RIGHT → action[left]=1 → dx=-1 → vx=-1*(-1)*speed = +speed (вправо). ✓
            left  = 1.0 if keys[pygame.K_RIGHT] else 0.0
            right = 1.0 if keys[pygame.K_LEFT] else 0.0
            kick  = 1.0 if keys[pygame.K_RETURN] else 0.0

        return torch.tensor(
            [up, down, left, right, kick],
            dtype=torch.float32,
            device=Constants.device
        )
