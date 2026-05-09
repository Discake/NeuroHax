import math
import torch
from Core.Domain.Entities.Map import Map
import Constants
from Player_actions.Net_action import Net_action


class Environment:
    """
    Среда для тренировки с простой системой наград
    Награда выдаётся за удары по мячу (касания)
    """
    
    def __init__(self, nn1, nn2, num_steps=2048):
        self.num_steps = num_steps
        self.nn1 = nn1
        self.nn2 = nn2
        self.map = Map()

        self.count = 0

        self.net_action_team1 = Net_action(self.map, nn1, self.map.players_team1[0], is_team_1=True)
        self.net_action_team2 = Net_action(self.map, nn2, self.map.players_team2[0], is_team_1=False)
        
        # Параметры наград - СИСТЕМА ДЛЯ ПИНКА МЯЧА (v5 - ДООБУЧЕНИЕ)
        # Ключевое: большая награда за УДАР для дообучения
        self.KICK_REWARD = 200.0            # ОЧЕНЬ большая награда за удар
        self.BALL_ACCEL_REWARD = 50.0       # Большая награда за ускорение мяча
        self.APPROACH_REWARD = 1.0          # Награда за приближение (меньше чем за удар)
        self.NEAR_BALL_REWARD = 0.5         # Награда за нахождение рядом

        # Пороги
        self.KICK_RANGE = 40.0              # Расстояние для удара
        self.NEAR_BALL_RANGE = 80.0         # Расстояние "рядом" с мячом
        self.BALL_ACCEL_THRESHOLD = 0.5     # Минимальное ускорение мяча
        
        # НЕТ возврата мяча - модель должна учиться преследовать!
        
        # Для отслеживания ударов
        self.last_kick_team1 = False
        self.last_kick_team2 = False
        
        # Для отслеживания ускорения мяча
        self.last_ball_vx = 0.0
        self.last_ball_vy = 0.0

    def step(self, action1, action2):
        """
        Шаг в среде
        
        Args:
            action1: Действие игрока 1
            action2: Действие игрока 2
            
        Returns:
            (state1, state2), (reward1, reward2), done, info
        """
        # Применяем действия
        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)

        # Двигаем объекты
        self.map.move_balls()

        self.count += 1

        # Получаем награды
        r1, r2, natural_done = self.compute_rewards()

        # === ОПРЕДЕЛЯЕМ truncated ===
        truncated = self.count >= self.num_steps

        # Эпизод завершён, если естественный конец ИЛИ таймаут
        done = natural_done or truncated

        info = {'truncated': truncated, 'natural_done': natural_done}

        # Получаем состояния
        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2), (r1, r2), done, info

    def reset(self):
        """Сброс среды для нового эпизода"""
        self.count = 0
        self.score_team1 = 0
        self.score_team2 = 0

        self.map.load_random()

        # Сбрасываем отслеживание ударов
        self.last_kick_team1 = False
        self.last_kick_team2 = False

        # Сбрасываем отслеживание мяча
        self.last_ball_vx = 0.0
        self.last_ball_vy = 0.0
        
        # Инициализируем расстояния до мяча
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]
        self.last_dist1 = math.sqrt((ball.x - player1.x)**2 + (ball.y - player1.y)**2)
        self.last_dist2 = math.sqrt((ball.x - player2.x)**2 + (ball.y - player2.y)**2)

        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2)

    def compute_rewards(self):
        """
        СИСТЕМА НАГРАД ДЛЯ ПИНКА МЯЧА (ОПТИМИЗИРОВАНО v2):
        1. Награда за ПРИБЛИЖЕНИЕ к мячу (ГЛАВНАЯ - учит двигаться к мячу)
        2. Награда за ускорение мяча (поощряет пинок)
        3. Награда за удар по мячу (большая)
        4. Награда за нахождение рядом (маленькая)
        """
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        r_team1 = 0.0
        r_team2 = 0.0
        done = False

        # Вычисляем расстояния до мяча
        dx1 = ball.x - player1.x
        dy1 = ball.y - player1.y
        dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)

        dx2 = ball.x - player2.x
        dy2 = ball.y - player2.y
        dist2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

        # === 1. НАГРАДА ЗА ПРИБЛИЖЕНИЕ К МЯЧУ (ГЛАВНАЯ!) ===
        # Игрок 1
        if dist1 < self.last_dist1:
            r_team1 += (self.last_dist1 - dist1) * self.APPROACH_REWARD
        self.last_dist1 = dist1

        # Игрок 2
        if dist2 < self.last_dist2:
            r_team2 += (self.last_dist2 - dist2) * self.APPROACH_REWARD
        self.last_dist2 = dist2

        # === 2. НАГРАДА ЗА УСКОРЕНИЕ МЯЧА ===
        ball_accel_x = ball.vx - self.last_ball_vx
        ball_accel_y = ball.vy - self.last_ball_vy
        ball_accel = math.sqrt(ball_accel_x * ball_accel_x + ball_accel_y * ball_accel_y)

        if ball_accel > self.BALL_ACCEL_THRESHOLD:
            if dist1 < dist2:
                r_team1 += ball_accel * self.BALL_ACCEL_REWARD
            else:
                r_team2 += ball_accel * self.BALL_ACCEL_REWARD

        self.last_ball_vx = ball.vx
        self.last_ball_vy = ball.vy

        # === 3. НАГРАДА ЗА НАХОЖДЕНИЕ РЯДОМ ===
        if dist1 <= self.NEAR_BALL_RANGE:
            proximity_reward = (1 - dist1 / self.NEAR_BALL_RANGE) * self.NEAR_BALL_REWARD
            r_team1 += proximity_reward

        if dist2 <= self.NEAR_BALL_RANGE:
            proximity_reward = (1 - dist2 / self.NEAR_BALL_RANGE) * self.NEAR_BALL_REWARD
            r_team2 += proximity_reward

        # === 4. НАГРАДА ЗА УДАР ===
        current_kick_team1 = player1.is_kicking
        current_kick_team2 = player2.is_kicking

        if current_kick_team1 and not self.last_kick_team1:
            if dist1 <= self.KICK_RANGE:
                r_team1 += self.KICK_REWARD

        if current_kick_team2 and not self.last_kick_team2:
            if dist2 <= self.KICK_RANGE:
                r_team2 += self.KICK_REWARD

        self.last_kick_team1 = current_kick_team1
        self.last_kick_team2 = current_kick_team2

        # Гол завершает эпизод
        if self.map.score_team1 or self.map.score_team2:
            done = True
            return torch.tensor([r_team1]), torch.tensor([r_team2]), done

        return torch.tensor([r_team1]), torch.tensor([r_team2]), done





                    
                