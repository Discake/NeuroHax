import math

import torch

from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Map import Map
from Core.Domain.Entities.Player import Player
from Vector2 import Vector2


class Rewards:
    def __init__(self, map : Map):
        self.map = map

        self.player1 = self.map.players_team1[0]
        self.player2 = self.map.players_team2[0]
        self.ball = self.map.balls[0]

        self._init_constants()

    def _init_constants(self):
        # === СИСТЕМА НАГРАД (упрощённая) ===
        self.VELOCITY_TOWARD_BALL_REWARD = 0.001  # Проекция скорости игрока на вектор к мячу
        self.INACTIVITY_PENALTY = -0.1         # Штраф за скорость ниже порога (снижен: -0.5 слишком жёстко)

        # Эллиптический (Гауссов) потенциал позиции мяча.
        # Центр эллипса — центр ворот соперника. Максимум = COEF на линии ворот по центру.
        self.BALL_POTENTIAL_COEF = 10

        self.GOAL_REWARD = 100.0
        self.GOAL_CONCEDED_PENALTY = -1000.0
        self.OWN_GOAL_PENALTY = -100.0
        # self.OWN_GOAL_PENALTY = -100.0

        # Награда за удар в створ ворот (прямой или с отскоком от борта).
        # Начисляется в момент удара, не зависит от скорости мяча.
        # Удары мимо створа не оцениваются.
        self.SHOT_ON_TARGET_REWARD = 0.005
        self.SHOT_ON_TARGET_PENALTY = -0.00  # штраф команде, по чьим воротам бьют
        # self.SHOT_ON_TARGET_PENALTY = -0.005  # штраф команде, по чьим воротам бьют

        # Бонус к награде за гол, если атакующий игрок обошёл соперника
        # (т.е. оказался ближе к воротам соперника, чем соперник к своим воротам).
        self.GOAL_POSITION_BONUS = 0#3000.0

        self.OPPONENT_IS_BEHIND_YOU_PENALTY = -0.0

        # Пороги
        self.KICK_RANGE = 35.0

        # Предвычисленные константы
        self._goal_cy = (self.map.goal_y_min + self.map.goal_y_max) / 2.0
        self._goal_half_h = (self.map.goal_y_max - self.map.goal_y_min) / 2.0
        self._inactivity_speed_sq = 10.0 ** 2
        self._goal_y_min = self.map.goal_y_min
        self._goal_y_max = self.map.goal_y_max
        self._field_width = self.map.width
        self._field_height = self.map.height
        self._half_x = self.map.width / 2.0
        # Параметры эллиптического потенциала
        # sigma_x = field_width (600): exp(-1)≈0.37 у своих ворот, 1.0 на линии ворот соперника
        # sigma_y = 1*goal_half_h (75): exp(-1)≈0.37 у края ворот, сужение вдвое относительно 2*goal_half_h
        self._sigma_x = float(self._field_width) / 2   # 600 px
        self._sigma_y = self._goal_half_h * 1    # 75 px
        # Базовое значение гауссиана на центральной линии (x = field_width/2).
        # Вычитается из потенциала, чтобы он был = 0 на центральной линии,
        # > 0 на половине соперника и < 0 на своей половине.
        self._potential_baseline = math.exp(-(self._half_x / self._sigma_x) ** 2)
        # Опорная скорость мяча для нормализации потенциала (типичная скорость после удара).
        # При |v_ball| >= _ball_speed_ref множитель = 1.0; при v=0 множитель = 0.
        self._ball_speed_ref = 100.0               # px/тик

        # Для отслеживания
        self.last_kick_team1 = False
        self.last_kick_team2 = False
        self.last_kicker = None  # 'team1' / 'team2' — кто последний бил
        self.ADVANCE_COEF = 0
        # self.ADVANCE_COEF = 1 / self._field_width
    
    def _is_shot_on_target_right(self, bx, by, vx, vy):
        """Летит ли мяч в правые ворота (x=W) — прямо или с одним отскоком от верхней/нижней стены."""
        if vx <= 0:
            return False
        W = self._field_width
        H = self._field_height
        g0, g1 = self._goal_y_min, self._goal_y_max

        # Прямой удар
        t = (W - bx) / vx
        if g0 <= by + vy * t <= g1:
            return True

        # Отскок от верхней стены (y = 0), мяч летит вверх (vy < 0)
        if vy < 0:
            t1 = -by / vy
            x1 = bx + vx * t1
            if bx < x1 < W:  # отскок происходит до линии ворот
                y_hit = (-vy) * (W - x1) / vx  # от y=0 с отражённой vy
                if g0 <= y_hit <= g1:
                    return True

        # Отскок от нижней стены (y = H), мяч летит вниз (vy > 0)
        if vy > 0:
            t1 = (H - by) / vy
            x1 = bx + vx * t1
            if bx < x1 < W:
                y_hit = H + (-vy) * (W - x1) / vx  # от y=H с отражённой vy
                if g0 <= y_hit <= g1:
                    return True

        return False
    
    def _is_shot_on_target_left(self, bx, by, vx, vy):
        """Летит ли мяч в левые ворота (x=0) — прямо или с одним отскоком от верхней/нижней стены."""
        if vx >= 0:
            return False
        H = self._field_height
        g0, g1 = self._goal_y_min, self._goal_y_max

        # Прямой удар
        t = -bx / vx  # vx < 0, bx > 0 → t > 0
        if g0 <= by + vy * t <= g1:
            return True

        # Отскок от верхней стены (y = 0), мяч летит вверх (vy < 0)
        if vy < 0:
            t1 = -by / vy
            x1 = bx + vx * t1
            if 0 < x1 < bx:  # отскок до линии ворот
                y_hit = (-vy) * (-x1 / vx)  # от y=0 с отражённой vy до x=0
                if g0 <= y_hit <= g1:
                    return True

        # Отскок от нижней стены (y = H), мяч летит вниз (vy > 0)
        if vy > 0:
            t1 = (H - by) / vy
            x1 = bx + vx * t1
            if 0 < x1 < bx:
                y_hit = H + (-vy) * (-x1 / vx)  # от y=H с отражённой vy до x=0
                if g0 <= y_hit <= g1:
                    return True

        return False

    def compute_rewards(self, step : int, prev_ball_x : float):
        self.r_team1 = 0.0
        self.r_team2 = 0.0

        self.prev_ball_x = prev_ball_x
        
        # self._add_dummy_potential()
        self._compute_deltas_and_vectors()

        self._add_velocity_towards_ball_reward()
        self._add_gaussian_potential_reward()
        self._add_kick_reward()
        self._add_inactivity_penalty()
        self._add_goal_reward()

        return torch.tensor([self.r_team1]), torch.tensor([self.r_team2]), self.done, self.new_kick1, self.new_kick2, self.goal_scored_team1, self.goal_scored_team2

    def _compute_deltas_and_vectors(self):
        player1 = self.player1
        player2 = self.player2
        ball = self.ball
        
        dx1 = ball.x - player1.x
        dy1 = ball.y - player1.y

        dx2 = ball.x - player2.x
        dy2 = ball.y - player2.y

        self._dx1, self._dx2 = dx1, dx2
        self._dy1, self._dy2 = dy1, dy2

        self._ball_pos_vec = Vector2(ball.x, ball.y)
        self._player1_pos_vec = Vector2(player1.x, player1.y)
        self._player2_pos_vec = Vector2(player2.x, player2.y)
        self._player1_vel_vec = Vector2(player1.vx, player1.vy)
        self._player2_vel_vec = Vector2(player2.vx, player2.vy)
        self._dist_delta1 = Vector2(dx1, dy1)
        self._dist_delta2 = Vector2(dx1, dx2)

    def _add_dummy_potential(self):
        # Для Team1 (атакует вправо)
        dx_ball = self.ball.x - self.prev_ball_x   # изменение x мяча за шаг
        advance_reward = self.ADVANCE_COEF * max(0.0, dx_ball)  # только вперёд
        self.r_team1 += advance_reward
        # Для Team2
        advance_reward = self.ADVANCE_COEF * max(0.0, -dx_ball)  # только вперёд
        self.r_team2 += advance_reward

        self.prev_ball_x = self.ball.x

    def _add_velocity_towards_ball_reward(self):
        vtb1 = Vector2.dot(self._dist_delta1, self._player1_vel_vec) / self._dist_delta1.norm2()
        if vtb1 > 0:
            self.r_team1 += vtb1 * self.VELOCITY_TOWARD_BALL_REWARD
        vtb2 = Vector2.dot(self._dist_delta2, self._player2_vel_vec) / self._dist_delta2.norm2()
        if vtb2 > 0:
            self.r_team2 += vtb2 * self.VELOCITY_TOWARD_BALL_REWARD

    def _add_gaussian_potential_reward(self):
        # === 2. ЭЛЛИПТИЧЕСКИЙ ПОТЕНЦИАЛ ПОЗИЦИИ МЯЧА ===
        # Гауссова форма, центр — центр ворот соперника.
        # Изолинии образуют эллипсы: максимум на линии ворот по центру,
        # минимум в углах и на своей половине.
        # Умножаем на нормализованную скорость мяча: при неподвижном мяче = 0,
        # при скорости >= BALL_SPEED_REF потенциал достигает полного значения.
        dy_goal = self.ball.y - self._goal_cy  # отклонение мяча от центра ворот по y
        y_factor = math.exp(-(dy_goal / self._sigma_y) ** 2)  # общий для обеих команд

        # Team1 атакует правые ворота (x = field_width).
        # max(0, ...): потенциал только положительный — нет штрафа за мяч на своей половине.
        dx1 = self._field_width - self.ball.x
        # self.r_team1 += self.BALL_POTENTIAL_COEF * max(0.0, math.exp(-(dx1 / self._sigma_x) ** 2) - self._potential_baseline) * y_factor
        self.r_team1 += self.BALL_POTENTIAL_COEF * math.exp(-(dx1 / self._sigma_x) ** 2) - self._potential_baseline * y_factor

        # Team2 атакует левые ворота (x = 0)
        dx2 = self.ball.x
        # self.r_team2 += self.BALL_POTENTIAL_COEF * max(0.0, math.exp(-(dx2 / self._sigma_x) ** 2) - self._potential_baseline) * y_factor
        self.r_team2 += self.BALL_POTENTIAL_COEF * (math.exp(-(dx2 / self._sigma_x) ** 2) - self._potential_baseline) * y_factor

    def _add_kick_reward(self):
        player1 = self.player1
        player2 = self.player2
        ball = self.ball

        # === 3. ОТСЛЕЖИВАНИЕ УДАРА (только для статистики) ===
        current_kick_team1 = player1.is_kicking
        current_kick_team2 = player2.is_kicking
        self.new_kick1 = False
        self.new_kick2 = False

        if current_kick_team1 and not self.last_kick_team1 and self._dist_delta1.norm2() <= self.KICK_RANGE:
            self.new_kick1 = True
            self.last_kicker = 'team1'

        if current_kick_team2 and not self.last_kick_team2 and self._dist_delta2.norm2() <= self.KICK_RANGE:
            self.new_kick2 = True
            self.last_kicker = 'team2'

        self.last_kick_team1 = current_kick_team1
        self.last_kick_team2 = current_kick_team2

        # === 3б. НАГРАДА ЗА УДАР В СТВОР ВОРОТ ===
        # Проверяем траекторию мяча после удара: прямо в ворота или с отскоком от борта.
        # Удары мимо створа не оцениваются. Не зависит от скорости мяча.

        if self._is_shot_on_target_right(ball.x, ball.y, ball.vx, ball.vy):
            self.r_team1 += self.SHOT_ON_TARGET_REWARD
            self.r_team2 += self.SHOT_ON_TARGET_PENALTY
        if self._is_shot_on_target_left(ball.x, ball.y, ball.vx, ball.vy):
            self.r_team2 += self.SHOT_ON_TARGET_REWARD
            self.r_team1 += self.SHOT_ON_TARGET_PENALTY

    def _add_inactivity_penalty(self):
        player1 = self.player1
        player2 = self.player2
        
        # === 4. ШТРАФ ЗА ПАССИВНОСТЬ ===
        if player1.vx * player1.vx + player1.vy * player1.vy < self._inactivity_speed_sq:
            self.r_team1 += self.INACTIVITY_PENALTY
        if player2.vx * player2.vx + player2.vy * player2.vy < self._inactivity_speed_sq:
            self.r_team2 += self.INACTIVITY_PENALTY

    def _add_goal_reward(self):
        # === 5. ПРОВЕРКА ГОЛОВ ===
        goal = self.map.check_goal()

        self.goal_scored_team1 = False
        self.goal_scored_team2 = False

        autogoal = False
        if goal == "team1":
            # Мяч в воротах team2 (правые ворота)
            self.goal_scored_team1 = True
            if self.last_kicker == 'team2':
                self.r_team2 += self.GOAL_CONCEDED_PENALTY + self.OWN_GOAL_PENALTY
                self.r_team1 += self.GOAL_REWARD
                autogoal = True
            else:
                # Бонус: player1 обошёл player2 (ближе к правым воротам, чем player2)
                bonus = 0
                # bonus = self.GOAL_POSITION_BONUS if player1.x > player2.x else 0.0
                self.r_team1 += self.GOAL_REWARD + bonus
                self.r_team2 += self.GOAL_CONCEDED_PENALTY
            self.map.reset_after_goal()
            self.last_kicker = None

        elif goal == "team2":
            # Мяч в воротах team1 (левые ворота)
            self.goal_scored_team2 = True
            if self.last_kicker == 'team1':
                self.r_team1 += self.GOAL_CONCEDED_PENALTY + self.OWN_GOAL_PENALTY
                self.r_team2 += self.GOAL_REWARD
                autogoal = True
            else:
                # Бонус: player2 обошёл player1 (ближе к левым воротам, чем player1)
                bonus = 0
                # bonus = self.GOAL_POSITION_BONUS if player2.x < player1.x else 0.0
                self.r_team2 += self.GOAL_REWARD + bonus
                self.r_team1 += self.GOAL_CONCEDED_PENALTY
            self.map.reset_after_goal()
            self.last_kicker = None

        self.done = autogoal