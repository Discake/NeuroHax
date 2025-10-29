import torch
from Data_structure.Vector import Vector

class Validable_vector(Vector):
    def validate(self, magnitude):
        # if self.length_squared() > magnitude * magnitude:
        #     scale = magnitude / self.length()
        #     self.x = self.x * scale
        #     self.y = self.y * scale
        pass

    def __add__(self, other):
        return  Validable_vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Validable_vector(self.x - other.x, self.y - other.y)

    def __mul__(self, number):
        return Validable_vector(self.x * number, self.y * number)

    def __truediv__(self, num):
        return Validable_vector(self.x / num, self.y / num)

    def __str__(self):
        return f"X: {self.x}, Y: {self.y}"
    
    def length_squared(self):
        return self.x.square() + self.y.square()

    def length(self):
        return torch.sqrt(self.length_squared())
    
        