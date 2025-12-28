from dataclasses import dataclass

@dataclass
class PlayerInput():
    team_id : int = None
    player_id : int = None
    move_x : float = None
    move_y : float = None
    kick : bool = False