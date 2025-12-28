from Core.Domain.GameConfig import GameConfig
from Core.Domain.IDrawing import IDrawing
from Core.Presentation.GamePresentation.ExternalDrawingInterface import ExternalDrawingInterface
from Core.Domain.StateDTO import StateDTO
from Core.Presentation.GamePresentation.DrawingConfig import DrawingConfig

class DrawingImpl(IDrawing):
    def __init__(self, ext_drawing_interface : ExternalDrawingInterface, g_config : GameConfig, d_config : DrawingConfig):
        self.d_config = d_config
        self.g_config = g_config
        self.edi = ext_drawing_interface
        # pygame.init()
        # self.screen = pygame.display.set_mode(d_config.window_size)

    def draw_state(self, state):
        dto = StateDTO(state, self.g_config)

        self.draw_map()
        self.draw_ball(dto)
        self.draw_player(dto)
        self.edi.update()

    def draw_map(self):
        # self.screen.fill(self.d_config.colors.background_color)  # Fill the screen with the filling color
        self.edi.set_background_color(self.d_config.colors.background_color)

        g_config = self.g_config
        """Draw the map on the surface."""
        # Using draw.rect module of
        # pygame to draw the solid rectangle

        window_size = self.d_config.window_size

        x_field_diff = g_config.field.x_center - g_config.field.width / 2
        y_field_diff = g_config.field.y_center - g_config.field.height / 2

        # pygame.draw.rect(self.screen, self.d_config.colors.field_color, [x_field_diff, y_field_diff, g_config.field.width, g_config.field.width], 0)
        self.edi.draw_rectangle(self.d_config.colors.field_color, [x_field_diff, y_field_diff, g_config.field.width, g_config.field.height])
        for gate in g_config.gates:
            gate_pos = [gate.x_center, gate.y_center]
            gate_corners = [gate_pos[0] - gate.width / 2, gate_pos[1] - gate.height / 2,
                            gate.width, gate.height]
            # pygame.draw.rect(self.screen, self.d_config.colors.gate_color, gate_corners, 0)
            self.edi.draw_rectangle(self.d_config.colors.gate_color, gate_corners)

        for wall in g_config.general_walls:
            wall_start = [wall.start[0], wall.start[1]]
            wall_end = [wall.end[0], wall.end[1]]
            # pygame.draw.line(self.screen, self.d_config.colors.gate_color, wall_start, wall_end, 0)
            self.edi.draw_line(self.d_config.colors.gate_color, wall_start, wall_end)

    def draw_ball(self, state : StateDTO):
        """Draw a ball on the surface."""

        for ball in state.balls:
            circle_color = self.d_config.colors.ball_color
            # pygame.draw.circle(self.screen, circle_color, (ball.x, ball.y), ball.radius)
            self.edi.draw_circle(circle_color, (ball.x, ball.y), ball.radius)

    def draw_player(self, state : StateDTO):
        """Draw a ball on the surface."""

        for player in state.players_team1:
            circle_color = self.d_config.colors.player_kicking_color_team1 if player.is_kicking else self.d_config.colors.player_color_team1
            # pygame.draw.circle(self.screen, circle_color, (player.x, player.y), player.radius)
            self.edi.draw_circle(circle_color, (player.x, player.y), player.radius)

        for player in state.players_team2:
            circle_color = self.d_config.colors.player_kicking_color_team2 if player.is_kicking else self.d_config.colors.player_color_team2
            # pygame.draw.circle(self.screen, circle_color, (player.x, player.y), player.radius)
            self.edi.draw_circle(circle_color, (player.x, player.y), player.radius)

