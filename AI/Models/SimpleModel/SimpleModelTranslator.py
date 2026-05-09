"""
Адаптер для использования SimpleModel с Environment
"""
import numpy as np
import torch
import Constants
from AI.GameTranslation.TranslatorToPolicy import TranslatorToPolicy
from Core.Domain.Entities.Map import Map


class SimpleModelTranslator(TranslatorToPolicy):
    """
    Translator для SimpleModel, совместимый с Environment
    Использует ту же логику, что и AI.Translator.Translator
    """

    def __init__(self, map: Map, player, is_team_1):
        # Передаём None как g_config, т.к. не используем родительский класс полностью
        super().__init__(None)
        self.map = map
        self.player = player
        self.is_team_1 = is_team_1

        # Предвычисленные константы — не меняются между вызовами
        self._flip = 1.0 if is_team_1 else -1.0
        self._inv_fw2  = 2.0 / Constants.window_size[0]   # 1 / (field_width / 2)
        self._inv_fh2  = 2.0 / Constants.window_size[1]   # 1 / (field_height / 2)
        self._inv_ps   = 1.0 / Constants.max_player_speed
        self._inv_bs   = 1.0 / Constants.max_ball_speed
        self._cx       = float(Constants.x_center)
        self._cy       = float(Constants.y_center)
        # Ворота соперника в мировых координатах (постоянны для каждой команды)
        self._opp_goal_x = float(Constants.field_size[0]) if is_team_1 else 0.0
        self._opp_goal_y = float(Constants.y_center)
        # Нормированная цель-ворота: (opp_goal_x - cx) * inv_fw2 * flip — постоянна
        self._goal_x_norm = (self._opp_goal_x - self._cx) * self._inv_fw2 * self._flip

        # Переиспользуемый numpy-буфер (избегаем Python list → tensor каждый шаг)
        self._buf = np.zeros(19, dtype=np.float32)

    def get_state_dim(self) -> int:
        return 19

    def translate(self, state) -> torch.Tensor:
        """Транслирует состояние в тензор для SimpleModel."""
        map_ = self.map
        flip = self._flip
        inv_fw2, inv_fh2 = self._inv_fw2, self._inv_fh2
        inv_ps, inv_bs   = self._inv_ps, self._inv_bs
        cx, cy           = self._cx, self._cy

        if self.is_team_1:
            me  = map_.players_team1[0]
            opp = map_.players_team2[0]
        else:
            me  = map_.players_team2[0]
            opp = map_.players_team1[0]

        ball = map_.balls[0]

        # Позиции и скорости (с флипом по X для team2)
        me_px  = (me.x  - cx) * inv_fw2 * flip
        me_py  = (me.y  - cy) * inv_fh2
        me_vx  = me.vx  * inv_ps * flip
        me_vy  = me.vy  * inv_ps

        op_px  = (opp.x - cx) * inv_fw2 * flip
        op_py  = (opp.y - cy) * inv_fh2
        op_vx  = opp.vx * inv_ps * flip
        op_vy  = opp.vy * inv_ps

        bx = ball.x; by = ball.y
        b_px   = (bx  - cx) * inv_fw2 * flip
        b_py   = (by  - cy) * inv_fh2
        b_vx   = ball.vx * inv_bs * flip
        b_vy   = ball.vy * inv_bs

        # Проекция скорости игрока на направление к мячу (без sqrt при dist≈0)
        ddx = bx - me.x;  ddy = by - me.y
        dist_sq = ddx * ddx + ddy * ddy
        if dist_sq > 1e-12:
            inv_d = dist_sq ** -0.5
            vel_toward_ball = (me.vx * ddx + me.vy * ddy) * inv_d * inv_ps
        else:
            vel_toward_ball = 0.0

        # Вектор мяч → ворота соперника (нормированный)
        ball_to_goal_x = (self._opp_goal_x - bx) * inv_fw2 * flip
        ball_to_goal_y = (self._opp_goal_y - by) * inv_fh2

        # Заполняем буфер напрямую (без создания Python list)
        buf = self._buf
        buf[0]  = me_px;  buf[1]  = me_py
        buf[2]  = me_vx;  buf[3]  = me_vy
        buf[4]  = b_px - me_px;  buf[5]  = b_py - me_py   # rel ball pos
        buf[6]  = b_vx;   buf[7]  = b_vy
        buf[8]  = op_px - me_px;  buf[9]  = op_py - me_py  # rel opp pos
        buf[10] = op_vx;  buf[11] = op_vy
        buf[12] = vel_toward_ball
        buf[13] = ball_to_goal_x;  buf[14] = ball_to_goal_y
        buf[15] = b_px;   buf[16] = b_py
        buf[17] = op_px;  buf[18] = op_py

        return torch.tensor(buf, dtype=torch.float32, device=Constants.device)
    
    def get_state(self):
        """Удобный метод для получения текущего состояния"""
        return self.translate({})
