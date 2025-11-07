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
        """Правильная симметричная трансляция"""
        
        # Определяем себя и противника
        if self.is_team_1:
            my_player = self.map.players_team1[0]
            opponent_player = self.map.players_team2[0]
            flip_coords = False
        else:
            my_player = self.map.players_team2[0]
            opponent_player = self.map.players_team1[0]
            flip_coords = False  # Отражаем поле для team2
            # flip_coords = True  # Отражаем поле для team2
        
        ball = self.map.balls[0]
        
        # Нормализация по правильным размерам
        field_width = Constants.window_size[0]
        field_height = Constants.window_size[1]
        
        # === МОЯ ПОЗИЦИЯ И СКОРОСТЬ ===
        my_pos_x = (my_player.position[0] - Constants.x_center) / (field_width / 2)
        my_pos_y = (my_player.position[1] - Constants.y_center) / (field_height / 2)
        my_vel_x = my_player.velocity[0] / Constants.max_player_speed
        my_vel_y = my_player.velocity[1] / Constants.max_player_speed
        my_kicking = float(my_player.is_kicking)
        
        # === ПОЗИЦИЯ И СКОРОСТЬ ПРОТИВНИКА ===
        opp_pos_x = (opponent_player.position[0] - Constants.x_center) / (field_width / 2)
        opp_pos_y = (opponent_player.position[1] - Constants.y_center) / (field_height / 2)
        opp_vel_x = opponent_player.velocity[0] / Constants.max_player_speed
        opp_vel_y = opponent_player.velocity[1] / Constants.max_player_speed
        opp_kicking = float(opponent_player.is_kicking)
        
        # === ПОЗИЦИЯ И СКОРОСТЬ МЯЧА ===
        ball_pos_x = (ball.position[0] - Constants.x_center) / (field_width / 2)
        ball_pos_y = (ball.position[1] - Constants.y_center) / (field_height / 2)
        ball_vel_x = ball.velocity[0] / Constants.max_ball_speed
        ball_vel_y = ball.velocity[1] / Constants.max_ball_speed
        
        # === ОТРАЖЕНИЕ ДЛЯ TEAM2 (симметрия) ===
        if flip_coords:
            # Отражаем X-координаты и X-скорости
            my_pos_x *= -1
            my_vel_x *= -1
            opp_pos_x *= -1
            opp_vel_x *= -1
            ball_pos_x *= -1
            ball_vel_x *= -1
        
        # === СБОРКА СОСТОЯНИЯ ===
        state = torch.tensor([
            # Моя информация (5)
            my_pos_x, my_pos_y, my_vel_x, my_vel_y, my_kicking,
            # Информация противника (5) 
            opp_pos_x, opp_pos_y, opp_vel_x, opp_vel_y, opp_kicking,
            # Информация о мяче (4)
            ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y
        ], device=Constants.device)
        
        # === ПРОВЕРКА ДИАПАЗОНА ===
        out_of_range = torch.any((state > 1.1) | (state < -1.1))
        # if out_of_range:
            # print(f"error in input")
            # Принудительно обрезаем
            # state = torch.clamp(state, -1.0, 1.0)
        
        return state