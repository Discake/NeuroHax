import math

class Vector2:
    def __init__(self, x1 : float, x2 : float):
        self.x1 = x1
        self.x2 = x2

    def dot(self, vector2):
        return dot(self, vector2)
    
    def norm(self):
        return math.sqrt(self.norm2)

    def norm2(self):
        return self.x1 * self.x1 + self.x2 * self.x2 + 1e-8
    
def dot(vector1, vector2):
        return vector1.x1 * vector2.x2 + vector1.x1 * vector2.x2