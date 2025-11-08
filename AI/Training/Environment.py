import torch
from Core.Objects.Map import Map
import Constants
from Player_actions.Net_action import Net_action

class Environment:
    def __init__(self, nn1, nn2, num_steps = 1024):
        self.num_steps = num_steps
        self.nn1 = nn1
        self.nn2 = nn2
        self.map = Map()

        #TODO исправить перебор игроков

        self.count = 0

        self.net_action_team1 = Net_action(self.map, nn1, self.map.players_team1[0], is_team_1=True)
        self.net_action_team2 = Net_action(self.map, nn2, self.map.players_team2[0], is_team_1=False)

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

    def step(self, action1, action2):

        self.net_action_team1.act(action1)
        self.net_action_team2.act(action2)
    
        self.map.move_balls()

        self.count += 1

        r1, r2, done = self.calculate_rewards()

        if not done:
            done = self.count > (self.num_steps - 3)

        if done:
            print("DONE!")
        
        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2), (r1, r2), done

    def reset(self):
        self.count = 0

        self.kick_count_modifier_team1 = 1
        self.kick_count_modifier_team2 = 1

        self.map.load_random()

        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]

        d1 = player1.position - ball.position
        sqrt1 = torch.linalg.vector_norm(d1)
        dist1 = max(0,sqrt1 - player1.radius - ball.radius)
        
        d2 = player2.position - ball.position
        sqrt2 = torch.linalg.vector_norm(d2)
        dist2 = max(0,sqrt2 - player2.radius - ball.radius)

        self.prev_dist1 = dist1
        self.prev_dist2 = dist2

        s1 = self.net_action_team1.translator.translate_input()
        s2 = self.net_action_team2.translator.translate_input()

        return (s1, s2)
    
    def calculate_rewards(self):
        r_team1 = 0.0
        r_team2 = 0.0
        
        player1 = self.map.players_team1[0]
        player2 = self.map.players_team2[0]
        ball = self.map.balls[0]
        
        # === 1. НАГРАДЫ ЗА УДАРЫ (умеренные, не экстремальные) ===
        if self.map.kick_flag_team1:
            r_team1 += 10.0 * self.kick_count_modifier_team1 # Было 200!
            r_team2 -= 9.0 * self.kick_count_modifier_team1
            self.kick_count_modifier_team1 += 1
        
        if self.map.kick_flag_team2:
            r_team2 += 10.0 * self.kick_count_modifier_team2
            r_team1 -= 9.0 * self.kick_count_modifier_team2
            self.kick_count_modifier_team2 += 1
        
        # === 2. НАГРАДЫ ЗА ПРИБЛИЖЕНИЕ К МЯЧУ ===
        d1 = ball.position - player1.position
        sqrt1 = torch.linalg.vector_norm(d1)
        dist1 = max(0, sqrt1 - player1.radius - ball.radius)
        
        d2 = ball.position - player2.position
        sqrt2 = torch.linalg.vector_norm(d2)
        dist2 = max(0, sqrt2 - player2.radius - ball.radius)
        
        max_dist = Constants.window_size[0]
    
        # ИЗМЕНЕНО: Награда за близость, а не штраф за удаление!
        # proximity_bonus1 = (1.0 - dist1 / max_dist) * 3.0  # [0, 3]
        # proximity_bonus2 = (1.0 - dist2 / max_dist) * 3.0
        
        # r_team1 += proximity_bonus1
        # r_team2 += proximity_bonus2
        
        # === 3. НАГРАДЫ ЗА ПРАВИЛЬНОЕ НАПРАВЛЕНИЕ ДВИЖЕНИЯ ===
        # МАСШТАБИРУЕМ чтобы она была сопоставима с другими наградами
        player1_speed = player1.velocity.norm()
        if sqrt1 > 1e-8 and player1_speed > 1e-8:
            direction_alignment1 = torch.dot(
                player1.velocity / player1_speed, 
                d1 / sqrt1
            )
            r_team1 += direction_alignment1 * 2.0  # Масштаб [-2, +2]
            # НОВОЕ: Дополнительная награда за БЫСТРОЕ движение к мячу
            if direction_alignment1 > 0.7:  # Двигается к мячу
                r_team1 += player1_speed / Constants.max_player_speed * 1.5
        
        player2_speed = player2.velocity.norm()
        if sqrt2 > 1e-8 and player2_speed > 1e-8:
            direction_alignment2 = torch.dot(
                player2.velocity / player2_speed, 
                d2 / sqrt2
            )
            r_team2 += direction_alignment2 * 2.0
            # НОВОЕ: Дополнительная награда за БЫСТРОЕ движение к мячу
            if direction_alignment2 > 0.7:  # Двигается к мячу
                r_team1 += player2_speed / Constants.max_player_speed * 1.5
        
        # === 5. ШТРАФ ЗА СТОЯНИЕ НА МЕСТЕ (КРИТИЧНО!) ===
        if player1_speed < 2.0:  # Почти не двигается
            r_team1 -= 1.0  # ШТРАФ за пассивность!
            
        if player2_speed < 2.0:
            r_team2 -= 1.0
        
        # === 5. ШТРАФЫ ЗА ГРАНИЦЫ (умеренные) ===
        done = False
        
        if (player1.position[0] < 0 or player1.position[0] > Constants.window_size[0] or
            player1.position[1] < 0 or player1.position[1] > Constants.window_size[1]):
            done = True
            r_team1 = -50.0  # Было -500!
            r_team2 += 10.0  # Бонус противнику
        
        if (player2.position[0] < 0 or player2.position[0] > Constants.window_size[0] or
            player2.position[1] < 0 or player2.position[1] > Constants.window_size[1]):
            done = True
            r_team2 = -50.0
            r_team1 += 10.0
        
        return torch.tensor([r_team1]), torch.tensor([r_team2]), done        
                    
                