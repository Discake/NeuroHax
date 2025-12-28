from Core.Domain.Entities.Ball import Ball

class Gate():
    def __init__(self, width, height, x_center, y_center):
        self.width = width
        self.height = height
        self.x_center = x_center
        self.y_center = y_center

    def is_ball_in(self, ball: Ball):
        return ball.x > self.x_center - self.width / 2 and ball.x < self.x_center + self.width / 2 and \
            ball.y > self.y_center - self.height / 2 and ball.y < self.y_center + self.height / 2