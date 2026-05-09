"""
Среда для тренировки SimpleModel
Использует SimpleModelNetAction для преобразования бинарных действий в движения

ОБНОВЛЁННАЯ СИСТЕМА НАГРАД:
- Награда за скорость в направлении мяча
- Штраф за пассивность
- Штраф за выход за пределы поля
- Награда за приближение к мячу
- Награда за удар по мячу
"""

import math
import torch
from Core.Domain.Entities.Map import Map
from AI.Models.SimpleModel.SimpleModelNetAction import SimpleModelNetAction
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator


class SimpleModelEnvironment:
    """
    Среда для тренировки SimpleModel с бинарными действиями (4 направления + kick)
    """

    def __init__(self, policy1, policy2, num_steps=1024):
        """
        Инициализация среды
        
        Args:
            policy1: SimpleModel для команды 1
            policy2: SimpleModel для команды 2
            num_steps: Максимальное количество шагов
        """
        self.num_steps = num_steps
        self.policy1 = policy1
        self.policy2 = policy2
        self.map = Map()

        self.count = 0

        # Используем SimpleModelNetAction вместо обычного Net_action
        self.net_action_team1 = SimpleModelNetAction(
            self.map, policy1, self.map.players_team1[0], is_team_1=True
        )
        self.net_action_team2 = SimpleModelNetAction(
            self.map, policy2, self.map.players_team2[0], is_team_1=False
        )

        # === СИСТЕМА НАГРАД (упрощённая) ===
        self.VELOCITY_TOWARD_BALL_REWARD = 0.005  # Проекция скорости игрока на вектор к мячу
        self.INACTIVITY_PENALTY = -0.05         # Штраф за скорость ниже порога (снижен: -0.5 слишком жёстко)

        # Эллиптический (Гауссов) потенциал позиции мяча.
        # Центр эллипса — центр ворот соперника. Максимум = COEF на линии ворот по центру.
        self.BALL_POTENTIAL_COEF = 1

        self.GOAL_REWARD = 200.0
        self.GOAL_CONCEDED_PENALTY = -200.0
        self.OWN_GOAL_PENALTY = -200.0

        # Награда за удар в створ ворот (прямой или с отскоком от борта).
        # Начисляется в момент удара, не зависит от скорости мяча.
        # Удары мимо створа не оцениваются.
        self.SHOT_ON_TARGET_REWARD = 1.0
        self.SHOT_ON_TARGET_PENALTY = -1.0  # штраф команде, по чьим воротам бьют

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
        self._sigma_y = self._goal_half_h * 0.7    # 75 px
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

    def step(self, action1, action2):
        """
        Шаг в среде
        
        Args:
            action1: Действие игрока 1 [up, down, left, right, kick] (binary)
            action2: Действие игрока 2 [up, down, left, right, kick] (binary)
        
        Returns:
            (state1, state2), (reward1, reward2), done, info
        """
        # Применяем действия через SimpleModelNetAction
        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)

        # Двигаем объекты
        self.map.move_balls()

        self.count += 1

        # Получаем награды
        r1, r2, natural_done, kick1, kick2, goal1, goal2 = self.compute_rewards()

        # Определяем truncated
        truncated = self.count >= self.num_steps
        done = truncated or natural_done  # эпизод заканчивается на голе или max_steps

        info = {'truncated': truncated, 'natural_done': natural_done,
                'kick_team1': kick1, 'kick_team2': kick2,
                'goal_team1': goal1, 'goal_team2': goal2}

        # Получаем состояния через translator
        self.net_action_team1.translator.map = self.map
        self.net_action_team2.translator.map = self.map

        s1 = self.net_action_team1.translator.translate({})
        s2 = self.net_action_team2.translator.translate({})

        return (s1, s2), (r1, r2), done, info

    def reset(self):
        """Сброс среды для нового эпизода"""
        self.count = 0
        self.map.load_random()

        # Сбрасываем отслеживание
        self.last_kick_team1 = False
        self.last_kick_team2 = False
        self.last_kicker = None

        # Сбрасываем net actions
        self.net_action_team1.reset()
        self.net_action_team2.reset()

        # Получаем начальные состояния
        self.net_action_team1.translator.map = self.map
        self.net_action_team2.translator.map = self.map


        s1 = self.net_action_team1.translator.translate({})
        s2 = self.net_action_team2.translator.translate({})

        return (s1, s2)


    def compute_rewards(self):
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        r_team1 = 0.0
        r_team2 = 0.0

        # if player1.x > player2.x and ball.x < self.map.width / 2:
        #     r_team1 += self.OPPONENT_IS_BEHIND_YOU_PENALTY
        # if player2.x < player1.x and ball.x > self.map.width / 2:
        #     r_team2 += self.OPPONENT_IS_BEHIND_YOU_PENALTY

        # === 1. СКОРОСТЬ К МЯЧУ ===
        dx1 = ball.x - player1.x
        dy1 = ball.y - player1.y
        dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)

        dx2 = ball.x - player2.x
        dy2 = ball.y - player2.y
        dist2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        if dist1 > 0.1:
            vtb1 = (player1.vx * dx1 + player1.vy * dy1) / dist1
            if vtb1 > 0:
                r_team1 += vtb1 * self.VELOCITY_TOWARD_BALL_REWARD

        if dist2 > 0.1:
            vtb2 = (player2.vx * dx2 + player2.vy * dy2) / dist2
            if vtb2 > 0:
                r_team2 += vtb2 * self.VELOCITY_TOWARD_BALL_REWARD

        # === 2. ЭЛЛИПТИЧЕСКИЙ ПОТЕНЦИАЛ ПОЗИЦИИ МЯЧА ===
        # Гауссова форма, центр — центр ворот соперника.
        # Изолинии образуют эллипсы: максимум на линии ворот по центру,
        # минимум в углах и на своей половине.
        # Умножаем на нормализованную скорость мяча: при неподвижном мяче = 0,
        # при скорости >= BALL_SPEED_REF потенциал достигает полного значения.
        dy_goal = ball.y - self._goal_cy  # отклонение мяча от центра ворот по y
        y_factor = math.exp(-(dy_goal / self._sigma_y) ** 2)  # общий для обеих команд

        # Направленная скорость мяча к воротам соперника (только положительная составляющая).
        # Team1 атакует вправо (+x), team2 атакует влево (-x).
        # При vx < 0 (мяч летит к своим воротам) множитель = 0 → нет ни награды ни штрафа.
        # Это устраняет incentive к автоголу: не давим на агента когда мяч на своей половине.
        v_toward_goal1 = min(max(ball.vx, 0.0) / self._ball_speed_ref, 1.0)
        v_toward_goal2 = min(max(-ball.vx, 0.0) / self._ball_speed_ref, 1.0)

        # Team1 атакует правые ворота (x = field_width).
        # max(0, ...): потенциал только положительный — нет штрафа за мяч на своей половине.
        # Умножаем на направленную скорость: награда только когда мяч летит к воротам соперника.
        dx1 = self._field_width - ball.x
        r_team1 += self.BALL_POTENTIAL_COEF * max(0.0, math.exp(-(dx1 / self._sigma_x) ** 2) - self._potential_baseline) * y_factor * v_toward_goal1

        # Team2 атакует левые ворота (x = 0)
        dx2 = ball.x
        r_team2 += self.BALL_POTENTIAL_COEF * max(0.0, math.exp(-(dx2 / self._sigma_x) ** 2) - self._potential_baseline) * y_factor * v_toward_goal2

        # === 3. ОТСЛЕЖИВАНИЕ УДАРА (только для статистики) ===
        current_kick_team1 = player1.is_kicking
        current_kick_team2 = player2.is_kicking
        new_kick1 = False
        new_kick2 = False

        if current_kick_team1 and not self.last_kick_team1 and dist1 <= self.KICK_RANGE:
            new_kick1 = True
            self.last_kicker = 'team1'

        if current_kick_team2 and not self.last_kick_team2 and dist2 <= self.KICK_RANGE:
            new_kick2 = True
            self.last_kicker = 'team2'

        self.last_kick_team1 = current_kick_team1
        self.last_kick_team2 = current_kick_team2

        # === 3б. НАГРАДА ЗА УДАР В СТВОР ВОРОТ ===
        # Проверяем траекторию мяча после удара: прямо в ворота или с отскоком от борта.
        # Удары мимо створа не оцениваются. Не зависит от скорости мяча.
        if new_kick1:
            if self._is_shot_on_target_right(ball.x, ball.y, ball.vx, ball.vy):
                r_team1 += self.SHOT_ON_TARGET_REWARD
                r_team2 += self.SHOT_ON_TARGET_PENALTY

        if new_kick2:
            if self._is_shot_on_target_left(ball.x, ball.y, ball.vx, ball.vy):
                r_team2 += self.SHOT_ON_TARGET_REWARD
                r_team1 += self.SHOT_ON_TARGET_PENALTY

        # === 4. ШТРАФ ЗА ПАССИВНОСТЬ ===
        if player1.vx * player1.vx + player1.vy * player1.vy < self._inactivity_speed_sq:
            r_team1 += self.INACTIVITY_PENALTY
        if player2.vx * player2.vx + player2.vy * player2.vy < self._inactivity_speed_sq:
            r_team2 += self.INACTIVITY_PENALTY

        # === 5. ПРОВЕРКА ГОЛОВ ===
        goal = self.map.check_goal()

        goal_scored_team1 = False
        goal_scored_team2 = False

        if goal == "team1":
            # Мяч в воротах team2 (правые ворота)
            goal_scored_team1 = True
            if self.last_kicker == 'team2':
                r_team2 += self.GOAL_CONCEDED_PENALTY + self.OWN_GOAL_PENALTY
                r_team1 += self.GOAL_REWARD
            else:
                # Бонус: player1 обошёл player2 (ближе к правым воротам, чем player2)
                bonus = self.GOAL_POSITION_BONUS if player1.x > player2.x else 0.0
                r_team1 += self.GOAL_REWARD + bonus
                r_team2 += self.GOAL_CONCEDED_PENALTY
            self.map.reset_after_goal()
            self.last_kicker = None

        elif goal == "team2":
            # Мяч в воротах team1 (левые ворота)
            goal_scored_team2 = True
            if self.last_kicker == 'team1':
                r_team1 += self.GOAL_CONCEDED_PENALTY + self.OWN_GOAL_PENALTY
                r_team2 += self.GOAL_REWARD
            else:
                # Бонус: player2 обошёл player1 (ближе к левым воротам, чем player1)
                bonus = self.GOAL_POSITION_BONUS if player2.x < player1.x else 0.0
                r_team2 += self.GOAL_REWARD + bonus
                r_team1 += self.GOAL_CONCEDED_PENALTY
            self.map.reset_after_goal()
            self.last_kicker = None

        done = goal is not None

        return torch.tensor([r_team1]), torch.tensor([r_team2]), done, new_kick1, new_kick2, goal_scored_team1, goal_scored_team2


def create_simple_model_environment(num_steps=2048):
    """
    Фабричная функция для создания среды с новыми SimpleModel
    
    Returns:
        env, policy1, policy2
    """
    from AI.Models.SimpleModel.Policy import SimpleModel
    
    # Создаём карту для инициализации translators
    map_obj = Map()
    map_obj.load_random()
    
    # Создаём translators
    translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
    translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
    
    # Создаём модели
    policy1 = SimpleModel(translator1)
    policy2 = SimpleModel(translator2)
    
    # Создаём среду
    env = SimpleModelEnvironment(policy1, policy2, num_steps=num_steps)
    
    return env, policy1, policy2
