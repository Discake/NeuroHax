from AI.GameTranslation.TranslatorToPolicy import TranslatorToPolicy
from Core.Domain.StateDTO import StateDTO
import torch


class TranslatorToPolicyImpl(TranslatorToPolicy):
    def __init__(self, g_config, team_id, player_id):
        super().__init__(g_config)
        self.player_id = player_id
        self.team_id = team_id

    def get_state_dim(self):
        return self.g_config.balls.max_balls * 4 + \
            self.g_config.team_red.max_players * 4 + \
                self.g_config.team_blue.max_players * 4

    def normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def translate(self, state):
        dto = StateDTO(state, self.g_config)

        self_pos = self.get_self_position(dto, self.team_id, self.player_id)
        self_vel = self.get_self_velocity(dto, self.team_id, self.player_id)
        opponent_poses = self.get_opponent_positions(dto, self.team_id, self.player_id)
        opponent_vels = self.get_opponent_velocities(dto, self.team_id, self.player_id)
        ball_poses = self.get_ball_positions(dto)
        ball_vels = self.get_ball_velocities(dto)


        normalized_values = [
            self.normalize(self_pos.x, 0, self.g_config.map.width),
            self.normalize(self_pos.y, 0, self.g_config.map.height),
            self.normalize(self_vel.x, -self.g_config.physics.player_max_speed, self.g_config.physics.player_max_speed),
            self.normalize(self_vel.y, -self.g_config.physics.player_max_speed, self.g_config.physics.player_max_speed),
        ]

        for opponent_pos in opponent_poses:
            normalized_values.extend([
                self.normalize(opponent_pos.x, 0, self.g_config.map.width),
                self.normalize(opponent_pos.y, 0, self.g_config.map.height),
            ])

        for opponent_vel in opponent_vels:
            normalized_values.extend([
                self.normalize(opponent_vel.x, -self.g_config.physics.player_max_speed, self.g_config.physics.player_max_speed),
                self.normalize(opponent_vel.y, -self.g_config.physics.player_max_speed, self.g_config.physics.player_max_speed),
            ])

        for ball_pos in ball_poses:
            normalized_values.extend([
                self.normalize(ball_pos.x, 0, self.g_config.map.width),
                self.normalize(ball_pos.y, 0, self.g_config.map.height),
            ])

        for ball_vel in ball_vels:
            normalized_values.extend([
                self.normalize(ball_vel.x, -self.g_config.physics.ball_max_speed, self.g_config.physics.ball_max_speed),
                self.normalize(ball_vel.y, -self.g_config.physics.ball_max_speed, self.g_config.physics.ball_max_speed),
            ])

        return torch.tensor(normalized_values, dtype=torch.float32)

        


