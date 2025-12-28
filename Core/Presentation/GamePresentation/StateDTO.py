from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Player import Player
from Core.Domain.GameConfig import GameConfig


class StateDTO():
    def __init__(self, state : dict, g_config : GameConfig):
        balls = state["balls"]
        self.balls : list[Ball] = [Ball(ball["x"], ball["y"], g_config.physics.ball_radius, g_config.physics.ball_mass, ball["vx"], ball["vy"]) \
                                   for ball in balls]
        
        players1 = state["players_team1"]
        self.players_team1 : list[Player] = [Player(player['id'], player['kick'], player['x'], player['y'], g_config.physics.player_mass, g_config.physics.player_radius,\
                                                    player['vx'], player['vy'], g_config.physics.kick_radius) \
                                                    for player in players1]
        
        players2 = state["players_team2"]
        self.players_team2 : list[Player] = [Player(player['id'], player['kick'], player['x'], player['y'], g_config.physics.player_mass, g_config.physics.player_radius,\
                                                    player['vx'], player['vy'], g_config.physics.kick_radius) \
                                                    for player in players2]
        self.score : list[int] = [state["score"][0], state["score"][1]]