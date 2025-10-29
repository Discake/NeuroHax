import Constants
from Player_actions.Player_action import Player_action
import pygame

class Keydown_action(Player_action):
    def __init__(self):
        self.keys = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_LSHIFT]
        self.pressed_keys = [False, False, False, False, False]


    def set_event(self, event):
        self.event = event
        if self.event.type == pygame.KEYDOWN:
            self.key_pressed_action(self.event.key)
        if self.event.type == pygame.KEYUP:
            self.key_released_action(self.event.key)

    def key_pressed_action(self, key):
        for i in range(len(self.keys)):
            if key == self.keys[i]:
                self.pressed_keys[i] = True
                break

    def key_released_action(self, key):
        for i in range(len(self.keys)):
            if key == self.keys[i]:
                self.pressed_keys[i] = False
                break

    def act(self):
        input = [float(self.pressed_keys[1]) - float(self.pressed_keys[0]), float(self.pressed_keys[3]) - float(self.pressed_keys[2])]
        super().act(input)
        if self.pressed_keys[4]:
            self.player.set_kicking(True)
        else:
            self.player.set_kicking(False)
