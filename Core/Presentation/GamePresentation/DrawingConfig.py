from dataclasses import dataclass

@dataclass(frozen=True)
class Colors:
    background_color : tuple[int, int, int]
    field_color : tuple[int, int, int]
    gate_color : tuple[int, int, int]
    ball_color : tuple[int, int, int]
    player_color_team1 : tuple[int, int, int]
    player_color_team2 : tuple[int, int, int]
    player_kicking_color_team1 : tuple[int, int, int]
    player_kicking_color_team2 : tuple[int, int, int]

@dataclass(frozen=True)
class DrawingConfig:
    colors : Colors
    window_size : tuple[int, int]