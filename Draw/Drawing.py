import pygame
import Constants


class Drawing:
    
    def __init__(self, map):
        self.screen = pygame.display.set_mode(Constants.window_size)
        self.screen.fill(Constants.filling_color)  # Fill the screen with the filling color
        self.map = map
        self.is_env = True

    def draw(self):
        self.draw_map(self.map)
    
    def draw_ball(self, surface, circle_color, ball):
        """Draw a ball on the surface."""
        pos_pair = ball.position
        # if not self.is_env:
        #     pos_pair = ball.position[0].item(), ball.position[1].item()
        pygame.draw.circle(surface, circle_color, (pos_pair[0].detach().item(), pos_pair[1].detach().item()), ball.radius)

    
    def draw_map(self, map_obj):
        self.screen.fill(Constants.filling_color)  # Fill the screen with the filling color

        """Draw the map on the surface."""
        # Using draw.rect module of
        # pygame to draw the solid rectangle
        pygame.draw.rect(self.screen, Constants.field_color, [100, 100, map_obj.width, map_obj.height], 0)
        for gate in map_obj.gates:
            gate_pos = gate.position.detach().tolist()

            x = gate_pos[0]
            y = gate_pos[1] - gate.outer_height / 2
            pygame.draw.rect(self.screen, gate.outer_color, [x, y, gate.outer_width, gate.outer_height], 0)

            gate_pos2 = (gate.position - gate.add_pos).detach().tolist()
            x2 = gate_pos2[0]
            y2 = gate_pos2[1]
            pygame.draw.rect(self.screen, gate.inner_color, [x2, y2, gate.inner_width, gate.inner_height], 0)
        #     gate.draw()

        # for line in map_obj.walls:
        #     line.draw()


        for team in map_obj.ball_teams:
            for ball in team:
                self.draw_ball(self.screen, ball.color, ball)
        