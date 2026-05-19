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
from Rewards import Rewards


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

        self.rewards = Rewards(self.map)

        # Для отрисовки
        self._sigma_x = self.rewards._sigma_x   # 600 px
        self._sigma_y = self.rewards._sigma_y    # 75 px
        self._goal_cy = self.rewards._goal_cy
        self._field_width = self.rewards._field_width
        self._field_height = self.rewards._field_height

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
        r1, r2, natural_done, kick1, kick2, goal1, goal2 = self.rewards.compute_rewards(self.count, self.prev_ball_x)

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

        ball = self.map.balls[0]
        self.prev_ball_x = ball.x 

        return (s1, s2)


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
