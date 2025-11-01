import torch
from Core.Physics.WallCollision import WallCollision as Wall
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
        self.position = torch.tensor([gate_data.center_x, gate_data.center_y], device=Constants.device)
        self.add_pos = torch.tensor([gate_data.add_pos[0], gate_data.add_pos[1]], device=Constants.device)
        self.boundaries = list[Wall]()

    def set_additional_pos(self, add_pos):
        self.add_pos = torch.tensor([add_pos[0], add_pos[1]], device=Constants.device)        

    def add_boundaries(self):
        boundary1 = Wall(self.position[0] - self.add_pos[0], self.position[0] - self.add_pos[0] + self.inner_width, self.position[1] - self.add_pos[1], False)
        boundary2 = Wall(self.position[0] - self.add_pos[0], self.position[0] - self.add_pos[0] + self.inner_width, self.position[1] - self.add_pos[1] + self.inner_height, False)
        
        if self.is_left:
            boundary3 = Wall(self.position[1] - self.add_pos[1], self.position[1] - self.add_pos[1] + self.inner_height, self.position[0] - self.add_pos[0], True)
        else:
            boundary3 = Wall(self.position[1] - self.add_pos[1], self.position[1] - self.add_pos[1] + self.inner_height, self.position[0] - self.add_pos[0] + self.inner_width, True)

        self.boundaries.append(boundary1)
        self.boundaries.append(boundary2)
        self.boundaries.append(boundary3)
        