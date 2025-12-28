from Core.Presentation.GamePresentation.DrawingConfig import Colors, DrawingConfig


d_config_window_size = [1000, 800]


d_config_colors_background_color = 255, 255, 255
d_config_colors_field_color = 0, 255, 0
d_config_colors_ball_color = 120, 120, 120
d_config_colors_gate_color = 30, 30, 30
d_config_colors_player_color_team1 = 255, 0, 0
d_config_colors_player_color_team2 = 0, 0, 255
d_config_colors_player_kicking_color_team1 = 0, 255, 255
d_config_colors_player_kicking_color_team2 = 255, 255, 0

colors = Colors(d_config_colors_background_color, d_config_colors_field_color, d_config_colors_ball_color,
                d_config_colors_gate_color, d_config_colors_player_color_team1, d_config_colors_player_color_team2,
                d_config_colors_player_kicking_color_team1, d_config_colors_player_kicking_color_team2)

drawing_config = DrawingConfig(colors, d_config_window_size)