"""
Визуализация игры NeuroHax
Отрисовка поля, игроков и мяча с использованием pygame
"""

import pygame
import sys
import math
import numpy as np
import Constants


class GameVisualizer:
    """
    Визуализатор игрового процесса NeuroHax
    """
    
    def __init__(self, width=800, height=600, title="NeuroHax"):
        """
        Инициализация визуализатора
        
        Args:
            width: Ширина окна
            height: Высота окна
            title: Заголовок окна
        """
        pygame.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Цвета
        self.colors = {
            'background': (30, 30, 30),
            'field': (50, 50, 50),
            'field_lines': (100, 100, 100),
            'ball': (255, 255, 255),
            'player1': (255, 50, 50),      # Красный (team1)
            'player2': (50, 50, 255),      # Синий (team2)
            'goal': (50, 255, 50),
            'text': (255, 255, 255),
            'info': (200, 200, 200)
        }
        
        # Параметры отрисовки
        self.ball_radius = 7
        self.player_radius = 15
        self.goal_width = 150   # высота проёма ворот (px)
        self.goal_height = 35   # глубина клетки ворот (px), вдавлена внутрь поля

        # Смещение поля внутри окна (мировые координаты → экранные)
        import Constants as _C
        self.fx = _C.field_offset_x   # 60
        self.fy = _C.field_offset_y   # 30
        self.fw = _C.field_size[0]    # 800
        self.fh = _C.field_size[1]    # 600
        
        # Состояние
        self.running = True
        self.paused = False
        self.show_info = True
        self.show_vectors = False
        self.show_potential = False
        self._potential_surface = None   # предвычисленная тепловая карта потенциала
        self.human_controls_hint = None  # Строка с подсказкой для живого игрока
        
    def handle_events(self):
        """Обработка событий pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                
                if event.key == pygame.K_SPACE and self.human_controls_hint != 'team1':
                    self.paused = not self.paused

                if event.key == pygame.K_p:
                    self.paused = not self.paused
                
                if event.key == pygame.K_i:
                    self.show_info = not self.show_info
                
                if event.key == pygame.K_v:
                    self.show_vectors = not self.show_vectors

                if event.key == pygame.K_h:
                    self.show_potential = not self.show_potential

                if event.key == pygame.K_r:
                    return 'reset'
        
        return True
    
    def set_potential_params(self, sigma_x, sigma_y, goal_cy, field_w, field_h):
        """
        Предвычисляет тепловую карту эллиптического потенциала для обеих команд.
        Вызывать один раз после создания среды; результат кешируется как pygame.Surface.

        Красный канал  = потенциал Team1 (атака вправо).
        Синий канал    = потенциал Team2 (атака влево).
        """
        GRID = 6  # мировых пикселей на ячейку сетки
        gw = max(field_w // GRID, 1)
        gh = max(field_h // GRID, 1)

        # Мировые координаты сетки в порядке (width, height) для surfarray
        xs = np.linspace(0, field_w, gw, dtype=np.float32)
        ys = np.linspace(0, field_h, gh, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')  # (gw, gh)

        dy       = yy - goal_cy
        y_factor = np.exp(-(dy / sigma_y) ** 2)

        baseline = np.exp(-((field_w / 2) / sigma_x) ** 2)
        p1 = (np.exp(-((field_w - xx) / sigma_x) ** 2) - baseline) * y_factor  # Team1: > 0 у ворот соперника
        p2 = (np.exp(-(xx           / sigma_x) ** 2) - baseline) * y_factor    # Team2: > 0 у ворот соперника

        # Положительные значения → насыщенный цвет, отрицательные → тёмный
        p1_pos = np.clip( p1, 0, None)   # только положительная часть
        p1_neg = np.clip(-p1, 0, None)   # модуль отрицательной части
        p2_pos = np.clip( p2, 0, None)
        p2_neg = np.clip(-p2, 0, None)

        scale = 1.0 / max(float(np.abs(p1).max()), float(np.abs(p2).max()), 1e-6)

        surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
        rgb   = pygame.surfarray.pixels3d(surf)    # (gw, gh, 3)
        alpha = pygame.surfarray.pixels_alpha(surf)  # (gw, gh)

        # Красный  — team1 (положительный), тёмно-красный — team1 (отрицательный)
        # Синий    — team2 (положительный), тёмно-синий   — team2 (отрицательный)
        # Зелёный  — смешение при взаимном погашении (около центра)
        rgb[:, :, 0] = np.clip((p1_pos - p2_pos) * scale * 200 + 30, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip((1 - (p1_pos + p2_pos) * scale) * 20, 0, 40).astype(np.uint8)
        rgb[:, :, 2] = np.clip((p2_pos - p1_pos) * scale * 200 + 30, 0, 255).astype(np.uint8)
        alpha[:, :]  = np.clip((p1_pos + p2_pos + p1_neg + p2_neg) * scale * 130 + 30, 30, 140).astype(np.uint8)

        del rgb, alpha  # снимаем блокировки surfarray

        self._potential_surface = pygame.transform.scale(surf, (self.fw, self.fh))

    def draw_potential_heatmap(self):
        """Рисует предвычисленную тепловую карту потенциала поверх поля."""
        if self._potential_surface is not None:
            self.screen.blit(self._potential_surface, (self.fx, self.fy))

    def draw_field(self):
        """Отрисовка поля с воротами. Поле смещено на (fx, fy) внутри окна."""
        fx, fy, fw, fh = self.fx, self.fy, self.fw, self.fh

        # Фон окна (за пределами поля)
        self.screen.fill(self.colors['background'])

        # Заливка поля
        pygame.draw.rect(self.screen, self.colors['field'], (fx, fy, fw, fh))

        goal_h    = self.goal_width
        goal_ymin = fy + (fh - goal_h) // 2
        goal_ymax = goal_ymin + goal_h
        depth     = self.goal_height
        post_r    = 8

        goal_c = self.colors['goal']
        wall_c = self.colors['field_lines']

        # ── Зоны ворот (ВНЕ поля, в отступе окна) ────────────────────
        goal_bg = (42, 52, 42)
        pygame.draw.rect(self.screen, goal_bg, (fx - depth, goal_ymin, depth, goal_h))       # левые
        pygame.draw.rect(self.screen, goal_bg, (fx + fw,    goal_ymin, depth, goal_h))       # правые

        # ── Граница поля (с проёмами ворот) ───────────────────────────
        pygame.draw.line(self.screen, wall_c, (fx, fy),       (fx + fw, fy),       3)  # верх
        pygame.draw.line(self.screen, wall_c, (fx, fy + fh),  (fx + fw, fy + fh),  3)  # низ
        pygame.draw.line(self.screen, wall_c, (fx, fy),        (fx, goal_ymin),     3)  # лево-верх
        pygame.draw.line(self.screen, wall_c, (fx, goal_ymax), (fx, fy + fh),       3)  # лево-низ
        pygame.draw.line(self.screen, wall_c, (fx + fw, fy),        (fx + fw, goal_ymin), 3)  # право-верх
        pygame.draw.line(self.screen, wall_c, (fx + fw, goal_ymax), (fx + fw, fy + fh),   3)  # право-низ

        # ── Клетка левых ворот (уходит влево от поля) ─────────────────
        pygame.draw.line(self.screen, goal_c, (fx, goal_ymin), (fx - depth, goal_ymin), 3)   # верхняя перекладина
        pygame.draw.line(self.screen, goal_c, (fx, goal_ymax), (fx - depth, goal_ymax), 3)   # нижняя перекладина
        pygame.draw.line(self.screen, goal_c, (fx - depth, goal_ymin), (fx - depth, goal_ymax), 3)  # задняя стенка
        pygame.draw.circle(self.screen, goal_c, (fx, goal_ymin), post_r)
        pygame.draw.circle(self.screen, goal_c, (fx, goal_ymax), post_r)

        # ── Клетка правых ворот (уходит вправо от поля) ───────────────
        pygame.draw.line(self.screen, goal_c, (fx + fw, goal_ymin), (fx + fw + depth, goal_ymin), 3)  # верхняя перекладина
        pygame.draw.line(self.screen, goal_c, (fx + fw, goal_ymax), (fx + fw + depth, goal_ymax), 3)  # нижняя перекладина
        pygame.draw.line(self.screen, goal_c, (fx + fw + depth, goal_ymin), (fx + fw + depth, goal_ymax), 3)  # задняя стенка
        pygame.draw.circle(self.screen, goal_c, (fx + fw, goal_ymin), post_r)
        pygame.draw.circle(self.screen, goal_c, (fx + fw, goal_ymax), post_r)

        # ── Центральная линия и круг ──────────────────────────────────
        cx = fx + fw // 2
        cy = fy + fh // 2
        pygame.draw.line(self.screen, wall_c, (cx, fy), (cx, fy + fh), 2)
        pygame.draw.circle(self.screen, wall_c, (cx, cy), 50, 2)
        pygame.draw.circle(self.screen, wall_c, (cx, cy), 5)

        # ── Штрафные зоны ─────────────────────────────────────────────
        pen_w, pen_h = 100, 250
        pen_y = fy + (fh - pen_h) // 2
        pygame.draw.rect(self.screen, wall_c, (fx,              pen_y, pen_w, pen_h), 2)
        pygame.draw.rect(self.screen, wall_c, (fx + fw - pen_w, pen_y, pen_w, pen_h), 2)
    
    def draw_ball(self, x, y, vx=0, vy=0):
        """
        Отрисовка мяча
        
        Args:
            x, y: Позиция мяча
            vx, vy: Скорость мяча (для вектора)
        """
        # Тень
        shadow_rect = pygame.Rect(x - self.ball_radius, y - self.ball_radius + 3,
                                  self.ball_radius * 2, self.ball_radius * 2)
        pygame.draw.ellipse(self.screen, (20, 20, 20), shadow_rect)
        
        # Мяч
        ball_rect = pygame.Rect(x - self.ball_radius, y - self.ball_radius,
                               self.ball_radius * 2, self.ball_radius * 2)
        pygame.draw.ellipse(self.screen, self.colors['ball'], ball_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), ball_rect, 2)
        
        # Вектор скорости
        if self.show_vectors and (abs(vx) > 0.1 or abs(vy) > 0.1):
            end_x = x + vx * 10
            end_y = y + vy * 10
            pygame.draw.line(self.screen, (255, 255, 0),
                           (x, y), (end_x, end_y), 2)
    
    def draw_player(self, x, y, team=1, vx=0, vy=0, is_kicking=False):
        """
        Отрисовка игрока
        
        Args:
            x, y: Позиция игрока
            team: Номер команды (1 или 2)
            vx, vy: Скорость игрока
            is_kicking: Флаг удара
        """
        color = self.colors['player1'] if team == 1 else self.colors['player2']
        
        # Тень
        shadow_rect = pygame.Rect(x - self.player_radius, y - self.player_radius + 3,
                                  self.player_radius * 2, self.player_radius * 2)
        pygame.draw.ellipse(self.screen, (20, 20, 20), shadow_rect)
        
        # Игрок
        player_rect = pygame.Rect(x - self.player_radius, y - self.player_radius,
                                 self.player_radius * 2, self.player_radius * 2)
        pygame.draw.ellipse(self.screen, color, player_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), player_rect, 2)
        
        # Индикатор удара
        if is_kicking:
            kick_rect = pygame.Rect(x - self.player_radius - 3, y - self.player_radius - 3,
                                   self.player_radius * 2 + 6, self.player_radius * 2 + 6)
            pygame.draw.ellipse(self.screen, (255, 255, 0), kick_rect, 2)
        
        # Вектор скорости
        if self.show_vectors and (abs(vx) > 0.1 or abs(vy) > 0.1):
            end_x = x + vx * 10
            end_y = y + vy * 10
            pygame.draw.line(self.screen, (255, 255, 0),
                           (x, y), (end_x, end_y), 2)
        
        # Номер команды (точка внутри)
        center_x = int(x)
        center_y = int(y)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), 4)
    
    def draw_score(self, score1, score2):
        """Отрисовка счёта"""
        score_text = f"{score1} : {score2}"
        text_surface = self.font.render(score_text, True, self.colors['text'])
        text_rect = text_surface.get_rect(center=(self.width // 2, 30))
        self.screen.blit(text_surface, text_rect)
        
        # Подписи команд
        team1_text = "Team 1"
        team2_text = "Team 2"
        team1_surface = self.small_font.render(team1_text, True, self.colors['player1'])
        team2_surface = self.small_font.render(team2_text, True, self.colors['player2'])
        
        self.screen.blit(team1_surface, (self.width // 2 - 100, 30))
        self.screen.blit(team2_surface, (self.width // 2 + 60, 30))
    
    def draw_info(self, episode, step, reward, fps):
        """Отрисовка информации"""
        if not self.show_info:
            return
        
        if self.human_controls_hint == 'team1':
            controls_lines = [
                "P - Pause/Resume",
                "WASD - Move  SPACE - Kick",
                "I - Info  V - Vectors  R - Reset  ESC - Exit",
            ]
        elif self.human_controls_hint == 'team2':
            controls_lines = [
                "P - Pause/Resume",
                "Arrows - Move  ENTER - Kick",
                "I - Info  V - Vectors  R - Reset  ESC - Exit",
            ]
        else:
            controls_lines = [
                "SPACE - Pause/Resume",
                "I - Info  V - Vectors  H - Heatmap  R - Reset  ESC - Exit",
            ]

        info_lines = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"FPS: {fps:.1f}",
            f"Paused: {'Yes' if self.paused else 'No'}",
            "",
            "Controls:",
        ] + controls_lines
        
        y_offset = 10
        for line in info_lines:
            color = self.colors['info']
            if line.startswith("Paused:"):
                color = self.colors['goal'] if self.paused else self.colors['text']
            
            text_surface = self.small_font.render(line, True, color)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 18
    
    def draw(self, map_obj, score1, score2, episode=0, step=0, reward=0.0):
        """
        Отрисовка всего состояния игры
        
        Args:
            map_obj: Объект карты с игроками и мячами
            score1, score2: Счёт команд
            episode: Номер эпизода
            step: Номер шага
            reward: Текущая награда
        """
        # Очистка и поле
        self.draw_field()

        # Тепловая карта потенциала (H для включения/выключения)
        if self.show_potential:
            self.draw_potential_heatmap()

        # Мяч (мировые координаты → экранные через смещение поля)
        if hasattr(map_obj, 'balls') and len(map_obj.balls) > 0:
            ball = map_obj.balls[0]
            self.draw_ball(ball.x + self.fx, ball.y + self.fy, ball.vx, ball.vy)

        # Игроки команды 1
        if hasattr(map_obj, 'players_team1'):
            for player in map_obj.players_team1:
                self.draw_player(player.x + self.fx, player.y + self.fy, team=1,
                                 vx=player.vx, vy=player.vy,
                                 is_kicking=player.is_kicking)

        # Игроки команды 2
        if hasattr(map_obj, 'players_team2'):
            for player in map_obj.players_team2:
                self.draw_player(player.x + self.fx, player.y + self.fy, team=2,
                                 vx=player.vx, vy=player.vy,
                                 is_kicking=player.is_kicking)
        
        # Счёт
        self.draw_score(score1, score2)
        
        # Информация
        fps = self.clock.get_fps()
        self.draw_info(episode, step, reward, fps)
        
        # Обновление экрана
        pygame.display.flip()
        
        # Обработка событий
        return self.handle_events()
    
    def close(self):
        """Закрытие визуализатора"""
        pygame.quit()
    
    def wait_for_start(self):
        """Ожидание начала игры"""
        waiting = True
        while waiting:
            self.screen.fill(self.colors['background'])
            
            # Заголовок
            title_font = pygame.font.Font(None, 48)
            title_text = title_font.render("NeuroHax", True, self.colors['text'])
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(title_text, title_rect)
            
            # Инструкция
            start_font = pygame.font.Font(None, 32)
            start_text = start_font.render("Press SPACE to start", True, self.colors['goal'])
            start_rect = start_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
            self.screen.blit(start_text, start_rect)
            
            # Управление
            controls = [
                "SPACE - Start/Pause",
                "ESC - Exit"
            ]
            
            y_offset = self.height // 2 + 80
            for line in controls:
                text_surface = self.small_font.render(line, True, self.colors['info'])
                text_rect = text_surface.get_rect(center=(self.width // 2, y_offset))
                self.screen.blit(text_surface, text_rect)
                y_offset += 25
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        waiting = False
            
            self.clock.tick(60)
        
        return True
