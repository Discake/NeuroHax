"""
Среда дообучения на голевые моменты.

Отличия от SimpleModelEnvironment:
- reset() использует load_for_goal_practice(difficulty)
- Убраны отвлекающие награды (приближение, удар ради удара)
- Добавлен штраф за нереализованный момент (timeout_penalty)
- difficulty управляется извне — курикулум в скрипте тренировки
"""

import math
import torch

from Core.Domain.Entities.Map import Map
from AI.Models.SimpleModel.SimpleModelNetAction import SimpleModelNetAction


class SimpleModelFineTuneEnv:
    def __init__(self, policy1, policy2, num_steps: int = 256,
                 difficulty: float = 0.0):
        self.num_steps  = num_steps
        self.difficulty = difficulty
        self.count      = 0

        self.map = Map()
        self.map.load_for_goal_practice(self.difficulty)

        self.net_action_team1 = SimpleModelNetAction(self.map, policy1,
                                                     self.map.players_team1[0], is_team_1=True)
        self.net_action_team2 = SimpleModelNetAction(self.map, policy2,
                                                     self.map.players_team2[0], is_team_1=False)

        # ── Награды (фокус: гол + базовые навыки) ───────────────────
        self.GOAL_REWARD          =  6000.0
        self.GOAL_CONCEDED        = -500.0
        self.OWN_GOAL_PENALTY     = -500.0

        # Плотные направляющие сигналы
        self.APPROACH_REWARD      =  10    # приближение к мячу
        self.KICK_REWARD          =  0.0   # совпадает с основной средой
        self.BALL_IN_GOAL_ZONE    =  2.0    # направляет мяч к воротам (не exploit при малом значении)
        self.DIRECTED_KICK_REWARD =  0.0
        self.BALL_TO_GOAL_REWARD  =  0.0
        self.BEHIND_BALL_REWARD   =  0   # без этого агент бьёт с любой стороны → мяч летит куда попало
        self.BALL_ACCEL_REWARD    =  0.0    # отключено (зависит от is_kicking)

        self.GOAL_ZONE_DEPTH      = 50
        self.INACTIVITY_PENALTY   = -0.03    # слабый (не провоцирует орбитальное движение)
        self.PASS_PENALTY         = 0.0   # штраф за потерю владения (пас сопернику)
        self.OUT_OF_BOUNDS_PENALTY = -3000.0  # огромный штраф + завершение эпизода
        self.OUT_OF_BOUNDS_BONUS   =   200.0  # бонус сопернику при выходе противника

        # Трекинг
        self.last_kick_team1 = False
        self.last_kick_team2 = False
        self.last_ball_speed = 0.0
        self.last_kicker     = None
        self.last_ball_owner = None   # 'team1', 'team2', None
        self.last_dist1      = 0.0
        self.last_dist2      = 0.0
        self.KICK_RANGE      = 50.0  # совпадает с kick_range в SimpleModelNetAction
        self.BALL_ACCEL_THR  = 0.0

    # ------------------------------------------------------------------ #

    def reset(self):
        self.count = 0
        self.map.load_for_goal_practice(self.difficulty, attacker=None)  # случайный атакующий

        self.last_kick_team1 = False
        self.last_kick_team2 = False
        self.last_ball_speed = 0.0
        self.last_kicker     = None
        self.last_ball_owner = None

        self.net_action_team1.reset()
        self.net_action_team2.reset()

        self.net_action_team1.translator.map = self.map
        self.net_action_team2.translator.map = self.map

        ball    = self.map.balls[0]
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        import math
        self.last_dist1 = math.sqrt((ball.x - player1.x)**2 + (ball.y - player1.y)**2)
        self.last_dist2 = math.sqrt((ball.x - player2.x)**2 + (ball.y - player2.y)**2)

        s1 = self.net_action_team1.translator.translate({})
        s2 = self.net_action_team2.translator.translate({})
        return s1, s2

    def step(self, action1, action2):
        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)
        self.map.move_balls()
        self.count += 1

        r1, r2, nat_done = self._compute_rewards()

        truncated = self.count >= self.num_steps
        done      = truncated or nat_done

        info = {'truncated': truncated, 'natural_done': nat_done}

        self.net_action_team1.translator.map = self.map
        self.net_action_team2.translator.map = self.map
        s1 = self.net_action_team1.translator.translate({})
        s2 = self.net_action_team2.translator.translate({})
        return (s1, s2), (r1, r2), done, info

    # ------------------------------------------------------------------ #

    def _compute_rewards(self):
        r1 = 0.0
        r2 = 0.0

        ball    = self.map.balls[0]
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]

        dx1  = ball.x - player1.x
        dy1  = ball.y - player1.y
        dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1) + 1e-6

        dx2  = ball.x - player2.x
        dy2  = ball.y - player2.y
        dist2 = math.sqrt(dx2 * dx2 + dy2 * dy2) + 1e-6

        ball_speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)

        # ── Смена владения мячом (пас сопернику) ─────────────────────
        near1 = dist1 <= self.KICK_RANGE
        near2 = dist2 <= self.KICK_RANGE
        if near1 and not near2:
            if self.last_ball_owner == 'team2':
                r2 += self.PASS_PENALTY   # team2 потеряла мяч → штраф team2
            self.last_ball_owner = 'team1'
        elif near2 and not near1:
            if self.last_ball_owner == 'team1':
                r1 += self.PASS_PENALTY   # team1 потеряла мяч → штраф team1
            self.last_ball_owner = 'team2'

        # ── Приближение к мячу ────────────────────────────────────────
        r1 += (self.last_dist1 - dist1) * self.APPROACH_REWARD
        self.last_dist1 = dist1
        r2 += (self.last_dist2 - dist2) * self.APPROACH_REWARD
        self.last_dist2 = dist2

        # ── Позиция "позади мяча" ─────────────────────────────────────
        if dist1 <= self.KICK_RANGE * 2:
            behind1 = (ball.x - player1.x) / (self.KICK_RANGE * 2)
            if behind1 > 0:
                r1 += behind1 * self.BEHIND_BALL_REWARD

        if dist2 <= self.KICK_RANGE * 2:
            behind2 = (player2.x - ball.x) / (self.KICK_RANGE * 2)
            if behind2 > 0:
                r2 += behind2 * self.BEHIND_BALL_REWARD

        # ── Удар ─────────────────────────────────────────────────────
        cur_kick1 = player1.is_kicking
        cur_kick2 = player2.is_kicking

        if cur_kick1 and not self.last_kick_team1 and dist1 <= self.KICK_RANGE:
            r1 += self.KICK_REWARD
            r1 += max(0.0, ball.vx) * self.DIRECTED_KICK_REWARD
            ball_accel = ball_speed - self.last_ball_speed
            if ball_accel > self.BALL_ACCEL_THR:
                r1 += ball_accel * self.BALL_ACCEL_REWARD
            self.last_kicker = 'team1'

        if cur_kick2 and not self.last_kick_team2 and dist2 <= self.KICK_RANGE:
            r2 += self.KICK_REWARD
            r2 += max(0.0, -ball.vx) * self.DIRECTED_KICK_REWARD
            ball_accel = ball_speed - self.last_ball_speed
            if ball_accel > self.BALL_ACCEL_THR:
                r2 += ball_accel * self.BALL_ACCEL_REWARD
            self.last_kicker = 'team2'

        self.last_ball_speed = ball_speed
        self.last_kick_team1 = cur_kick1
        self.last_kick_team2 = cur_kick2

        # ── Штраф за пассивность ─────────────────────────────────────
        speed1 = math.sqrt(player1.vx ** 2 + player1.vy ** 2)
        speed2 = math.sqrt(player2.vx ** 2 + player2.vy ** 2)
        if speed1 < 20.0:
            r1 += self.INACTIVITY_PENALTY
        if speed2 < 20.0:
            r2 += self.INACTIVITY_PENALTY

        # ── Штраф за выход за поле (завершает эпизод) ────────────────
        w, h = self.map.width, self.map.height
        rad1, rad2 = player1.radius, player2.radius
        oob1 = player1.x < rad1 or player1.x > w - rad1 or player1.y < rad1 or player1.y > h - rad1
        oob2 = player2.x < rad2 or player2.x > w - rad2 or player2.y < rad2 or player2.y > h - rad2
        # if oob1:
        #     r1 += self.OUT_OF_BOUNDS_PENALTY
        #     r2 += self.OUT_OF_BOUNDS_BONUS
        # if oob2:
        #     r2 += self.OUT_OF_BOUNDS_PENALTY
        #     r1 += self.OUT_OF_BOUNDS_BONUS
        # oob_done = oob1 or oob2
        oob_done = False

        # ── Мяч движется к воротам ────────────────────────────────────
        r1 += max(0.0,  ball.vx) * self.BALL_TO_GOAL_REWARD
        r2 += max(0.0, -ball.vx) * self.BALL_TO_GOAL_REWARD

        # ── Воронка к воротам ─────────────────────────────────────────
        gmin = self.map.goal_y_min
        gmax = self.map.goal_y_max
        zone = self.GOAL_ZONE_DEPTH

        if gmin <= ball.y <= gmax and ball.x > self.map.width - zone:
            r1 += (ball.x - (self.map.width - zone)) / zone * self.BALL_IN_GOAL_ZONE

        if gmin <= ball.y <= gmax and ball.x < zone:
            r2 += (zone - ball.x) / zone * self.BALL_IN_GOAL_ZONE

        # ── Гол ──────────────────────────────────────────────────────
        goal = self.map.check_goal()

        if goal == 'team1':
            if self.last_kicker == 'team2':
                r2 += self.GOAL_CONCEDED + self.OWN_GOAL_PENALTY
                r1 += self.GOAL_REWARD
            else:
                r1 += self.GOAL_REWARD
                r2 += self.GOAL_CONCEDED
            self.map.load_for_goal_practice(self.difficulty, attacker=None)
            self.last_kicker = None
            self.last_kick_team1 = False
            self.last_kick_team2 = False
            self.last_ball_speed = 0.0
            self.last_ball_owner = None
            ball2 = self.map.balls[0]
            p1 = self.map.players_team1[0]
            p2 = self.map.players_team2[0]
            self.last_dist1 = math.sqrt((ball2.x - p1.x)**2 + (ball2.y - p1.y)**2)
            self.last_dist2 = math.sqrt((ball2.x - p2.x)**2 + (ball2.y - p2.y)**2)

        elif goal == 'team2':
            if self.last_kicker == 'team1':
                r1 += self.GOAL_CONCEDED + self.OWN_GOAL_PENALTY
                r2 += self.GOAL_REWARD
            else:
                r2 += self.GOAL_REWARD
                r1 += self.GOAL_CONCEDED
            self.map.load_for_goal_practice(self.difficulty, attacker=None)
            self.last_kicker = None
            self.last_kick_team1 = False
            self.last_kick_team2 = False
            self.last_ball_speed = 0.0
            self.last_ball_owner = None
            ball2 = self.map.balls[0]
            p1 = self.map.players_team1[0]
            p2 = self.map.players_team2[0]
            self.last_dist1 = math.sqrt((ball2.x - p1.x)**2 + (ball2.y - p1.y)**2)
            self.last_dist2 = math.sqrt((ball2.x - p2.x)**2 + (ball2.y - p2.y)**2)

        done = (goal is not None) or oob_done

        return torch.tensor([r1]), torch.tensor([r2]), False
