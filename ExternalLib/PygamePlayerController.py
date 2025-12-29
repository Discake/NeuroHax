import pygame

from Core.Domain.PlayerInput import PlayerInput
from Core.Infrastructure.PlayerController import PlayerController

class PygamePlayerController(PlayerController):
    def __init__(self, player_id, team_id):
        super().__init__(player_id, team_id)
        self.keys = [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_LSHIFT]
        self.pressed_keys = [0 for _ in range(len(self.keys))]
        self.event = None
        self.output = None

    def check_events(self):
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.KEYDOWN:
                self.event = event
                self.output = self.key_pressed_action(self.event.key)
                return
            if event.type == pygame.KEYUP:
                self.event = event
                self.output = self.key_released_action(self.event.key)
                return
        
    def key_pressed_action(self, key):
        input = PlayerInput()
        input.player_id = self.player_id
        input.team_id = self.team_id

        for i in range(len(self.keys)):
            if key == self.keys[i]:
                self.pressed_keys[i] = 1
                break

        move_x = float(self.pressed_keys[1]) - float(self.pressed_keys[0])
        move_y = float(self.pressed_keys[3]) - float(self.pressed_keys[2])
        kick = int(self.pressed_keys[4])

        input.move_x = move_x
        input.move_y = move_y
        input.kick = kick
        
        return input

    def key_released_action(self, key):
        input = PlayerInput()
        input.player_id = self.player_id
        input.team_id = self.team_id
        
        for i in range(len(self.keys)):
            if key == self.keys[i]:
                self.pressed_keys[i] = 0
                break

        move_x = float(self.pressed_keys[1]) - float(self.pressed_keys[0])
        move_y = float(self.pressed_keys[3]) - float(self.pressed_keys[2])
        kick = int(self.pressed_keys[4])

        input.move_x = move_x
        input.move_y = move_y
        input.kick = kick
        
        return input

    def is_acting(self, state):
        self.check_events()

        return self.event is not None and (self.event.type == pygame.KEYUP or self.event.type == pygame.KEYDOWN)

    def get_action(self, state):
        return self.output


        