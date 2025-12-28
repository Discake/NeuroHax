import pygame

from Core.Presentation.GamePresentation.ExternalDrawingInterface import ExternalDrawingInterface

class PygameDrawingInterfaceImpl(ExternalDrawingInterface):
    def __init__(self, window_size, background_color):
        pygame.init()
        self.surface = self.set_window(window_size, background_color)

    def draw_rectangle(self, color, corners):
        pygame.draw.rect(self.surface, color, corners, 0)

    def draw_circle(self, color, center, radius):
        pygame.draw.circle(self.surface, color, center, radius)

    def draw_line(self, color, start, end):
        pygame.draw.line(self.surface, color, start, end, 3)

    def set_window(self, window_size, background_color):
        return pygame.display.set_mode(window_size)

    def set_background_color(self, color):
        self.surface.fill(color)

    def update(self):
        pygame.display.update()



    

    