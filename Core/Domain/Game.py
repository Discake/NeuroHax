from Core.Domain.GameConfig import GameConfig
from Core.Domain.AbstractGame import AbstractGame
from Core.Domain.Entities.Gate import Gate
from Core.Domain.Entities.Ball import Ball
from Core.Domain.Physics import Physics
from Core.Domain.Entities.Player import Player
from Core.Domain.Entities.Map import Map
from Core.Domain.Entities.Field import Field
from Core.Domain.Entities.Wall import Wall
from Core.Domain.IDrawing import IDrawing

class Game(AbstractGame):
    def __init__(self, id:str, config:GameConfig):
        self.id = id
        self.config = config
        self.balls : list[Ball] = []
        self.players_team1 : list[Player] = []
        self.players_team2 : list[Player] = []
        self.score = [0, 0]
        self.gates : list[Gate] = []
        self.general_walls : list[Wall] = []
        self.ball_walls : list[Wall] = []

        self.add_players_and_balls()
        self.add_map_and_field()
        self.add_gates_and_walls()

    def get_state(self):
        state = {
            "balls": [],
            "players_team1": [],
            "players_team2": [],
            "score": self.score
        }

        for ball in self.balls:
            state["balls"].append({"x": ball.x, "y": ball.y, "vx": ball.vx, "vy": ball.vy})

        for player in self.players_team1:
            state["players_team1"].append({"id": player.id, "x": player.x, "y": player.y, "vx": player.vx, "vy": player.vy, "kick": player.is_kicking})

        for player in self.players_team2:
            state["players_team2"].append({"id": player.id, "x": player.x, "y": player.y, "vx": player.vx, "vy": player.vy, "kick": player.is_kicking})

        return state
    
    def update(self, physics : Physics):
        dt = 1 / self.config.physics.tick_rate

        for ball in self.balls:
            ball.move(dt)
        for player in self.players_team1:
            player.move(dt)
        for player in self.players_team2:
            player.move(dt)

        all_balls = self.balls + self.players_team1 + self.players_team2
        all_players = self.players_team1 + self.players_team2
        for i in range(len(all_balls)):
            for j in range(i + 1, len(all_balls)):
                physics.resolve_collision_with_ball(all_balls[i], all_balls[j])

        for ball in all_balls:
            for wall in self.general_walls:
                physics.resolve_collision_with_wall(wall, ball)

        for ball in self.balls:
            for wall in self.ball_walls:
                physics.resolve_collision_with_wall(wall, ball)

        for ball in self.balls:
            physics.restrict_velocity(ball, self.config.physics.ball_max_speed)
            physics.apply_friction(ball, dt)

        for player in all_players:
            physics.restrict_velocity(player, self.config.physics.player_max_speed)
            physics.apply_friction(player, dt)

        for player in all_players:
            for ball in self.balls:
                physics.kick_ball(player, ball, self.config.physics.kick_power)

        for gate in self.gates:
            for ball in self.balls:
                if gate.is_ball_in(ball):
                    if gate.x_center < self.field.x_center:
                        self.score[1] += 1
                    else:
                        self.score[0] += 1
                    self.load()

    def render(self, drawing : IDrawing):
        drawing.draw_state(self.get_state())

    def load(self):
        red_players_num = self.config.team_red.max_players
        blue_players_num = self.config.team_blue.max_players

        red_players_positions = self.config.team_red.players_positions
        blue_players_positions = self.config.team_blue.players_positions

        for i in range(self.config.balls.max_balls):
            x, y = self.config.balls.balls_positions[i]
            ball = self.balls[i]
            ball.x = x
            ball.y = y
            ball.vx = 0
            ball.vy = 0

        for i in range(red_players_num):
            x, y = red_players_positions[i]
            player = self.players_team1[i]
            player.x = x
            player.y = y
            player.vx = 0
            player.vy = 0

        for i in range(blue_players_num):
            x, y = blue_players_positions[i]
            player = self.players_team2[i]
            player.x = x
            player.y = y
            player.vx = 0
            player.vy = 0
  

    def add_players_and_balls(self):
        red_players_num = self.config.team_red.max_players
        blue_players_num = self.config.team_blue.max_players

        red_players_positions = self.config.team_red.players_positions
        blue_players_positions = self.config.team_blue.players_positions

        ball_radius = self.config.physics.ball_radius
        player_radius = self.config.physics.player_radius

        player_mass = self.config.physics.player_mass
        ball_mass = self.config.physics.ball_mass

        player_kick_radius = self.config.physics.kick_radius

        for i in range(red_players_num):
            x, y = red_players_positions[i]
            player = Player(i, False, x, y, player_mass, player_radius, 0, 0, player_kick_radius)
            self.players_team1.append(player)
            
        for i in range(blue_players_num):
            x, y = blue_players_positions[i]
            player = Player(i, False, x, y, player_mass, player_radius, 0, 0, player_kick_radius)
            self.players_team2.append(player)

        for i in range(self.config.balls.max_balls):
            x, y = self.config.balls.balls_positions[i]
            self.balls.append(Ball(x, y, ball_radius, ball_mass, 0, 0))

    def add_map_and_field(self):
        map_size = self.config.map
        self.map = Map(map_size.width, map_size.height)

        field_cfg = self.config.field
        self.field = Field(field_cfg.width, field_cfg.height, field_cfg.x_center, field_cfg.y_center)

    def add_gates_and_walls(self):
        gates = self.config.gates

        for gate in gates:
            self.gates.append(Gate(gate.width, gate.height, gate.x_center, gate.y_center))

        general_walls = self.config.general_walls
        ball_walls = self.config.ball_walls

        for wall in general_walls:
            self.general_walls.append(Wall(wall.start[0], wall.start[1], wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]))

        for wall in ball_walls:
            self.ball_walls.append(Wall(wall.start[0], wall.start[1], wall.end[0] - wall.start[0], wall.end[1] - wall.start[1]))

    def handle_input(self, input):
        team = self.players_team1 if input.team_id == 0 else self.players_team2
        player = team[input.player_id]

        player.vx += input.move_x * self.config.physics.player_move_modificator
        player.vy += input.move_y * self.config.physics.player_move_modificator
        player.is_kicking = input.kick

    def get_all_players_ids(self):
        return [player.id for player in self.players_team1], [player.id for player in self.players_team2]