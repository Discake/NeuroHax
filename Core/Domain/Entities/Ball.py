class Ball():
    def __init__(self, x, y, radius, mass, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass

    def move(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt