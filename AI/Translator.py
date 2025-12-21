import torch
import Constants
from Core.Objects.Map import Map

class Translator:
    def __init__(self, map: Map, net, player, is_team_1):
        self.map = map
        self.net = net
        self.input = torch.zeros(Constants.player_number * 4 + Constants.ball_number * 4 + 1)
        self.output = [0, 0, 0]
        self.player = player
        self.is_team_1 = is_team_1

    def translate_input(self):
        """Правильная симметричная трансляция с канонической целью"""
        
        # 1. Определяем, нужно ли отражать мир
        if self.is_team_1:
            my_player = self.map.players_team1[0]
            opponent_player = self.map.players_team2[0]
            flip = 1.0
        else:
            my_player = self.map.players_team2[0]
            opponent_player = self.map.players_team1[0]
            flip = -1.0 # Просто запоминаем, что нужно будет отразить X

        ball = self.map.balls[0]
        
        # 2. Нормализуем ВСЕ базовые координаты и скорости
        field_width = Constants.window_size[0]
        field_height = Constants.window_size[1]

        # Мои параметры (в глобальной системе координат)
        my_pos_x = (my_player.position[0] - Constants.x_center) / (field_width / 2)
        my_pos_y = (my_player.position[1] - Constants.y_center) / (field_height / 2)
        my_vel_x = my_player.velocity[0] / Constants.max_player_speed
        my_vel_y = my_player.velocity[1] / Constants.max_player_speed
        # my_kicking = float(my_player.is_kicking)

        # Параметры противника
        opp_pos_x = (opponent_player.position[0] - Constants.x_center) / (field_width / 2)
        opp_pos_y = (opponent_player.position[1] - Constants.y_center) / (field_height / 2)
        opp_vel_x = opponent_player.velocity[0] / Constants.max_player_speed
        opp_vel_y = opponent_player.velocity[1] / Constants.max_player_speed
        opp_kicking = float(opponent_player.is_kicking)

        # Параметры мяча
        ball_pos_x = (ball.position[0] - Constants.x_center) / (field_width / 2)
        ball_pos_y = (ball.position[1] - Constants.y_center) / (field_height / 2)
        ball_vel_x = ball.velocity[0] / Constants.max_ball_speed
        ball_vel_y = ball.velocity[1] / Constants.max_ball_speed

        ball_pos_x_with_gate = (ball.position[0] - Constants.x_center) / (Constants.field_size[0] / 2)
        ball_pos_y_with_gate = (ball.position[1] - Constants.y_center) / (Constants.field_size[1] / 2)

        # 3. ### ПРИМЕНЯЕМ ОТРАЖЕНИЕ ###
        # Умножаем все X-компоненты на `flip` (-1 для team2, 1 для team1)
        my_pos_x *= flip
        my_vel_x *= flip
        opp_pos_x *= flip
        opp_vel_x *= flip
        ball_pos_x *= flip
        ball_vel_x *= flip
        ball_pos_x_with_gate *= flip

        # 4. ### РАССЧИТЫВАЕМ ПРОИЗВОДНЫЕ ВЕЛИЧИНЫ ПОСЛЕ ОТРАЖЕНИЯ ###
        # Теперь мир "канонический", и цель ВСЕГДА справа.
        
        # Координаты цели АТАКИ всегда одинаковы в этой новой системе
        target_goal_x = 1.0 
        target_goal_y = 0.0

        # Вектор от МЯЧА до ворот (более полезная информация, чем от игрока)
        ball_to_goal_x = target_goal_x - ball_pos_x_with_gate
        ball_to_goal_y = target_goal_y - ball_pos_y_with_gate
        
        # Дистанция от мяча до ворот
        dist_ball_to_goal = (ball_to_goal_x**2 + ball_to_goal_y**2)**0.5
        
        # === НОВЫЕ ПОЛЕЗНЫЕ ПРИЗНАКИ ===
        # Относительная позиция мяча к игроку (важно для микроконтроля)
        rel_ball_pos_x = ball_pos_x - my_pos_x
        rel_ball_pos_y = ball_pos_y - my_pos_y
        
        # Относительная позиция противника
        rel_opp_pos_x = opp_pos_x - my_pos_x
        rel_opp_pos_y = opp_pos_y - my_pos_y

        # Добавить флаг "твоя сторона поля"
        my_side = 1.0 if (my_player.position[0] < Constants.x_center) == my_player.is_team1 else -1.0

        # 5. ### СОБИРАЕМ ИТОГОВЫЙ STATE ###
        # Используем только относительные и канонические координаты
        state = torch.tensor([
            # Моя позиция и скорость (абсолютная, но в канонической системе)
            my_pos_x, my_pos_y, my_vel_x, my_vel_y,
            
            # Позиция мяча (относительно меня)
            rel_ball_pos_x, rel_ball_pos_y,
            
            # Скорость мяча (абсолютная, в канонической системе)
            ball_vel_x, ball_vel_y,
            
            # Позиция противника (относительно меня)
            rel_opp_pos_x, rel_opp_pos_y, opp_vel_x, opp_vel_y,
            
            # Информация о цели (относительно мяча, это "подсказка", куда везти мяч)
            ball_to_goal_x, ball_to_goal_y, dist_ball_to_goal,

            my_side
            
            # Можно добавить еще признаки, если нужно, но БЕЗ side_flag
        ], device=Constants.device)
        
        return state
