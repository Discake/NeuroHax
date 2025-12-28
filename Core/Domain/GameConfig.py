from dataclasses import dataclass

@dataclass(frozen=True)
class TeamConfig:
    max_players: int
    players_positions: list[tuple[float, float]]

@dataclass(frozen=True)
class BallsConfig:
    max_balls: int
    balls_positions: list[tuple[float, float]]


@dataclass(frozen=True)
class PhysicsConfig:
    tick_rate: int          # тиков в секунду
    ball_radius: float
    player_radius: float
    ball_mass: float
    player_mass: float
    kick_radius: float
    kick_power: float
    friction: float         # трение
    player_max_speed: float
    ball_max_speed: float
    player_move_modificator: float


@dataclass(frozen=True)
class FieldConfig:
    width: float
    height: float
    x_center: float
    y_center: float

@dataclass(frozen=True)
class MapConfig:
    width: float
    height: float

@dataclass(frozen=True)
class GateConfig:
    width: float
    height: float
    x_center: float
    y_center: float

@dataclass(frozen=True)
class WallConfig:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True)
class GameConfig:
    map: MapConfig
    field: FieldConfig
    gates: list[GateConfig]
    physics: PhysicsConfig
    team_red: TeamConfig
    team_blue: TeamConfig
    balls: BallsConfig
    general_walls: list[WallConfig]
    ball_walls: list[WallConfig]
    