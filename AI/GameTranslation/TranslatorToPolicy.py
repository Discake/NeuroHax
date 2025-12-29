from abc import ABC, abstractmethod

from torch import Tensor

from AI.GameTranslation.VectorClasses import Position, Velocity
from Core.Domain.GameConfig import GameConfig
from Core.Domain.StateDTO import StateDTO

class TranslatorToPolicy(ABC):
    def __init__(self, g_config : GameConfig):
        self.g_config = g_config

    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    @abstractmethod
    def translate(self, state : dict) -> Tensor:
        pass

    def get_self_position(self, dto : StateDTO, self_team_id, self_player_id):
        self_team = dto.players_team1 if self_team_id == 0 else dto.players_team2

        for player in self_team:
            if player.id == self_player_id:
                return Position(player.x, player.y)
        raise Exception("Player not found")

    def get_self_velocity(self, dto : StateDTO, self_team_id, self_player_id):
        self_team = dto.players_team1 if self_team_id == 0 else dto.players_team2
        for player in self_team:
            if player.id == self_player_id:
                return Velocity(player.vx, player.vy)
        raise Exception("Player not found")

    def get_ball_positions(self, dto : StateDTO):
        return [Position(ball.x, ball.y) for ball in dto.balls]
    
    def get_ball_velocities(self, dto : StateDTO):
        return [Velocity(ball.vx, ball.vy) for ball in dto.balls]
    
    def get_opponent_positions(self, dto : StateDTO, self_team_id, self_player_id):
        moves : list[Position] = []
        opp_team = dto.players_team1 if self_team_id == 1 else dto.players_team2

        for player in opp_team:
            # if player.id != self_player_id:
                moves.append(Position(player.x, player.y))

        return moves

    def get_opponent_velocities(self, dto : StateDTO, self_team_id, self_player_id):
        moves : list[Velocity] = []
        opp_team = dto.players_team1 if self_team_id == 1 else dto.players_team2

        for player in opp_team:
            # if player.id != self_player_id:
                moves.append(Velocity(player.vx, player.vy))

        return moves
        