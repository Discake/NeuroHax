from abc import ABC, abstractmethod

class ExternalDrawingInterface(ABC):
    @abstractmethod
    def draw_rectangle(self, color, corners):
        pass
    @abstractmethod
    def draw_circle(self, color, center, radius):
        pass
    @abstractmethod
    def draw_line(self, color, start, end):
        pass
    @abstractmethod
    def set_window(self, window_size, background_color):
        pass
    @abstractmethod
    def set_background_color(self, color):
        pass
    @abstractmethod
    def update(self):
        pass