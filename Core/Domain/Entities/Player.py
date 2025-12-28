from Core.Domain.Entities.Ball import Ball

class Player(Ball):
    is_kicking : bool = False
    id : int = None

    def __init__(self, id, kick, x, y, mass, radius, vx, vy, kick_radius):
        super().__init__(x, y, radius, mass, vx, vy)
        self.kick_radius = kick_radius
        self.id = id
        self.is_kicking = kick