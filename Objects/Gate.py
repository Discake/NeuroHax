import torch
from Data_structure.Vector import Vector
from Physics.WallCollision import WallCollision as Wall
import Constants

class Gate:
    def __init__(self, gate_data):
        self.is_left = gate_data.is_left
        self.outer_height = Constants.gates_outer_height
        self.outer_width = Constants.gates_outer_width
        self.inner_height = Constants.gates_inner_height
        self.inner_width = Constants.gates_inner_width
        self.inner_color = Constants.gates_inner_color
        self.outer_color = Constants.gates_outer_color
        self.position = Vector(torch.tensor(gate_data.center_x, dtype=torch.float32).to(Constants.device), torch.tensor(gate_data.center_y, dtype=torch.float32).to(Constants.device))
        self.add_pos = Vector(torch.tensor(gate_data.add_pos[0], dtype=torch.float32).to(Constants.device), torch.tensor(gate_data.add_pos[1], dtype=torch.float32).to(Constants.device))

        self.boundaries = []

    def set_additional_pos(self, add_pos):
        self.add_pos = Vector(add_pos[0], add_pos[1])        

    def add_boundaries(self):
        boundary1 = Wall(self.position.x - self.add_pos.x, self.position.x - self.add_pos.x + self.inner_width, self.position.y - self.add_pos.y, False)
        boundary2 = Wall(self.position.x - self.add_pos.x, self.position.x - self.add_pos.x + self.inner_width, self.position.y - self.add_pos.y + self.inner_height, False)
        
        if self.is_left:
            boundary3 = Wall(self.position.y - self.add_pos.y, self.position.y - self.add_pos.y + self.inner_height, self.position.x - self.add_pos.x, True)
        else:
            boundary3 = Wall(self.position.y - self.add_pos.y, self.position.y - self.add_pos.y + self.inner_height, self.position.x - self.add_pos.x + self.inner_width, True)

        self.boundaries.append(boundary1)
        self.boundaries.append(boundary2)
        self.boundaries.append(boundary3)

    def draw(self):
        for bound in self.boundaries:
            bound.draw()

    def set_screen(self, screen):
        self.screen = screen
        