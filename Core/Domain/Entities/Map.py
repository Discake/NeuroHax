"""
Map - Игровая карта с игроками и мячами
"""
import math
import random
import torch
import Constants
from Core.Domain.Entities.Ball import Ball
from Core.Domain.Entities.Player import Player
from Core.Domain.Entities.Field import Field
from Core.Infrastructure.Physics.PhysicsImpl import PhysicsImpl
from Core.Domain.GameConfig import PhysicsConfig


class Map():
    """
    Игровая карта содержащая игроков и мячи.

    Геометрия ворот:
      - GOAL_HEIGHT = 150px  — высота проёма ворот
      - GOAL_DEPTH  = 35px   — глубина клетки (вдавлена в поле для отрисовки)
      - Стойки (GOAL_POST_RADIUS = 8) в точках (0, goal_y_min/max) и (width, goal_y_min/max)

    Физика:
      - Мяч отскакивает от всех стен КРОМЕ проёма ворот; влетает в ворота → check_goal()
      - Игрок свободно ходит по полю и может войти в зону ворот,
        но не может пройти сквозь стойки
    """

    GOAL_HEIGHT     = 150
    GOAL_DEPTH      = 35   # визуальная глубина клетки (px), вдавлена внутрь поля
    GOAL_POST_RADIUS = 8

    def __init__(self, width=Constants.field_size[0], height=Constants.field_size[1]):
        self.width  = width
        self.height = height

        physics_config = PhysicsConfig(
            tick_rate=60,
            ball_radius=7,
            player_radius=15,
            ball_mass=0.5,
            player_mass=1.0,
            kick_radius=20,
            kick_power=200.0,
            friction=0.35,
            player_max_speed=Constants.max_player_speed,
            ball_max_speed=Constants.max_ball_speed,
            player_move_modificator=2.0
        )
        self.physics = PhysicsImpl(physics_config)

        self.players_team1 = []
        self.players_team2 = []
        self.balls = []

        self.score_team1 = 0
        self.score_team2 = 0

        self.field = Field(
            width=Constants.field_size[0],
            height=Constants.field_size[1],
            x_center=Constants.x_center,
            y_center=Constants.y_center
        )

        self._init_players_and_balls()
        # Кешированные списки — не меняются в течение эпизода
        self._all_players = self.players_team1 + self.players_team2
        self._all_objects = self.balls + self._all_players

    # ------------------------------------------------------------------ #
    #  Свойства ворот                                                      #
    # ------------------------------------------------------------------ #

    @property
    def goal_y_min(self):
        return (self.height - self.GOAL_HEIGHT) / 2   # 225

    @property
    def goal_y_max(self):
        return (self.height + self.GOAL_HEIGHT) / 2   # 375

    # ------------------------------------------------------------------ #
    #  Инициализация                                                       #
    # ------------------------------------------------------------------ #

    def _init_players_and_balls(self):
        player1 = Player(id=0, kick=False,
                         x=self.width * 0.25, y=self.height / 2,
                         mass=1.0, radius=15, vx=0, vy=0, kick_radius=20)
        player1.is_team1 = True
        self.players_team1.append(player1)

        player2 = Player(id=1, kick=False,
                         x=self.width * 0.75, y=self.height / 2,
                         mass=1.0, radius=15, vx=0, vy=0, kick_radius=20)
        player2.is_team1 = False
        self.players_team2.append(player2)

        ball = Ball(x=self.width / 2, y=self.height / 2,
                    radius=7, mass=1.0, vx=0, vy=0)
        self.balls.append(ball)

    def reset_after_goal(self):
        """Сброс позиций после гола — эпизод продолжается."""
        cx, cy = self.width / 2, self.height / 2

        self.players_team1[0].x = self.width * 0.25
        self.players_team1[0].y = cy
        self.players_team1[0].vx = 0;  self.players_team1[0].vy = 0
        self.players_team1[0].is_kicking = False

        self.players_team2[0].x = self.width * 0.75
        self.players_team2[0].y = cy
        self.players_team2[0].vx = 0;  self.players_team2[0].vy = 0
        self.players_team2[0].is_kicking = False

        self.balls[0].x = cx;  self.balls[0].y = cy
        self.balls[0].vx = 0;  self.balls[0].vy = 0

    def load_for_goal_practice(self, difficulty: float = 0.0, attacker: str = None):
        """
        Расстановка для отработки голевых моментов.

        difficulty=0.0 — лёгкий: мяч прямо перед воротами, игрок чётко позади.
        difficulty=1.0 — сложный: случайные позиции на атакующей половине.
        attacker='team1'|'team2'|None(random) — кто атакует.
        """
        if attacker is None:
            attacker = random.choice(['team1', 'team2'])

        goal_cy   = self.height / 2
        half_goal = self.GOAL_HEIGHT / 2 * 0.8
        max_dist  = 60 + difficulty * 220
        y_spread  = half_goal * difficulty

        ball_y = goal_cy + random.uniform(-y_spread, y_spread)
        ball_y = max(30.0, min(self.height - 30.0, ball_y))

        offset_x = random.uniform(40, 80 + difficulty * 60)
        offset_y = random.uniform(-15 * difficulty, 15 * difficulty)

        if attacker == 'team1':
            # team1 атакует правые ворота (x = width)
            ball_x = self.width - random.uniform(25, max(26, max_dist))
            att_x  = max(15.0, ball_x - offset_x)          # атакующий — левее мяча
            att_y  = max(15.0, min(self.height - 15.0, ball_y + offset_y))
            def_x  = self.width - random.uniform(15, 40 + difficulty * 80)
            def_y  = goal_cy + random.uniform(-half_goal, half_goal)
            def_y  = max(15.0, min(self.height - 15.0, def_y))
            self.players_team1[0].x = att_x;  self.players_team1[0].y = att_y
            self.players_team2[0].x = def_x;  self.players_team2[0].y = def_y
        else:
            # team2 атакует левые ворота (x = 0)
            ball_x = random.uniform(25, max(26, max_dist))
            att_x  = min(self.width - 15.0, ball_x + offset_x)  # атакующий — правее мяча
            att_y  = max(15.0, min(self.height - 15.0, ball_y + offset_y))
            def_x  = random.uniform(15, 40 + difficulty * 80)
            def_y  = goal_cy + random.uniform(-half_goal, half_goal)
            def_y  = max(15.0, min(self.height - 15.0, def_y))
            self.players_team2[0].x = att_x;  self.players_team2[0].y = att_y
            self.players_team1[0].x = def_x;  self.players_team1[0].y = def_y

        for p in (self.players_team1[0], self.players_team2[0]):
            p.vx = 0.0;  p.vy = 0.0;  p.is_kicking = False

        self.balls[0].x  = ball_x;  self.balls[0].y  = ball_y
        self.balls[0].vx = 0.0;     self.balls[0].vy = 0.0

    def load_random(self):
        """Случайная расстановка для нового эпизода (счёт не сбрасывается)."""
        self.players_team1[0].x  = random.uniform(50, self.width / 2 - 50)
        self.players_team1[0].y  = random.uniform(80, self.height - 80)
        self.players_team1[0].vx = 0;  self.players_team1[0].vy = 0
        self.players_team1[0].is_kicking = False

        self.players_team2[0].x  = random.uniform(self.width / 2 + 50, self.width - 50)
        self.players_team2[0].y  = random.uniform(80, self.height - 80)
        self.players_team2[0].vx = 0;  self.players_team2[0].vy = 0
        self.players_team2[0].is_kicking = False

        self.balls[0].x  = random.uniform(100, self.width - 100)
        self.balls[0].y  = random.uniform(80,  self.height - 80)
        self.balls[0].vx = 0;  self.balls[0].vy = 0

    # ------------------------------------------------------------------ #
    #  Физика                                                              #
    # ------------------------------------------------------------------ #

    def move_balls(self):
        """Перемещение всех объектов и применение физики за один тик."""
        dt = 1 / 60.0
        all_players = self._all_players
        all_objects = self._all_objects

        # 1. Перемещение
        for obj in all_objects:
            obj.move(dt)

        # 2. Столкновения между объектами
        for i in range(len(all_objects)):
            for j in range(i + 1, len(all_objects)):
                self.physics.resolve_collision_with_ball(all_objects[i], all_objects[j])

        # 3. Стены для МЯЧА (с учётом проёма ворот)
        gmin, gmax = self.goal_y_min, self.goal_y_max
        for ball in self.balls:
            # Верхняя/нижняя стены — всегда отскок
            if ball.y - ball.radius < 0:
                ball.y  = ball.radius
                ball.vy = abs(ball.vy) * 0.8
            elif ball.y + ball.radius > self.height:
                ball.y  = self.height - ball.radius
                ball.vy = -abs(ball.vy) * 0.8

            # Левая стена — отскок вне проёма ворот
            if ball.x - ball.radius < 0:
                if not (gmin <= ball.y <= gmax):
                    ball.x  = ball.radius
                    ball.vx = abs(ball.vx) * 0.8

            # Правая стена — отскок вне проёма ворот
            if ball.x + ball.radius > self.width:
                if not (gmin <= ball.y <= gmax):
                    ball.x  = self.width - ball.radius
                    ball.vx = -abs(ball.vx) * 0.8

        # 4. Границы для ИГРОКОВ (мягкое ограничение — окно, + рамка ворот)
        margin_x = Constants.field_offset_x   # 60 — горизонтальный отступ окна
        margin_y = Constants.field_offset_y   # 30 — вертикальный отступ окна
        for player in all_players:
            # Игрок не выходит за края ОКНА
            player.x = max(-margin_x + player.radius,
                           min(self.width  + margin_x - player.radius, player.x))
            player.y = max(-margin_y + player.radius,
                           min(self.height + margin_y - player.radius, player.y))

        # Стойки ворот — твёрдые круги на границе поля
        posts = [
            (0,          gmin),
            (0,          gmax),
            (self.width, gmin),
            (self.width, gmax),
        ]
        for player in all_players:
            for px, py in posts:
                self._resolve_post_collision(player, px, py)

        # Рамка ворот — сегменты ВНЕ поля (перекладины + задняя стенка)
        d = self.GOAL_DEPTH
        goal_segments = [
            # Левые ворота (уходят в -x)
            (0,           gmin,  -d,          gmin),   # верхняя перекладина
            (0,           gmax,  -d,          gmax),   # нижняя перекладина
            (-d,          gmin,  -d,          gmax),   # задняя стенка
            # Правые ворота (уходят за ширину поля)
            (self.width,  gmin,  self.width+d, gmin),  # верхняя перекладина
            (self.width,  gmax,  self.width+d, gmax),  # нижняя перекладина
            (self.width+d, gmin, self.width+d, gmax),  # задняя стенка
        ]
        # Сегменты ворот находятся за пределами поля — проверяем только если игрок рядом с воротами
        seg_margin = self.GOAL_DEPTH + 16  # GOAL_DEPTH(35) + player_radius(15) + запас
        for player in all_players:
            if player.x <= seg_margin or player.x >= self.width - seg_margin:
                for ax, ay, bx, by in goal_segments:
                    self._resolve_segment_collision(player, ax, ay, bx, by)

        # 5. Трение и ограничение скорости
        for ball in self.balls:
            self.physics.apply_friction(ball, dt)
            self.physics.restrict_velocity(ball, Constants.max_ball_speed)

        for player in all_players:
            self.physics.apply_friction(player, dt)
            self.physics.restrict_velocity(player, Constants.max_player_speed)

    def _resolve_post_collision(self, player, post_x, post_y):
        """Разрешение столкновения игрока с неподвижной стойкой ворот."""
        dx = player.x - post_x
        dy = player.y - post_y
        dist_sq = dx * dx + dy * dy
        min_dist = player.radius + self.GOAL_POST_RADIUS
        if dist_sq >= min_dist * min_dist or dist_sq == 0:
            return
        dist = math.sqrt(dist_sq)
        nx, ny = dx / dist, dy / dist
        # Вытолкнуть игрока из стойки
        overlap = min_dist - dist
        player.x += nx * overlap
        player.y += ny * overlap
        # Погасить составляющую скорости «внутрь» стойки
        vel_into = player.vx * nx + player.vy * ny
        if vel_into < 0:
            player.vx -= vel_into * nx
            player.vy -= vel_into * ny

    def _resolve_segment_collision(self, player, ax, ay, bx, by):
        """Разрешение столкновения игрока с отрезком (перекладина/задняя стенка ворот)."""
        dx = bx - ax
        dy = by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return
        t = ((player.x - ax) * dx + (player.y - ay) * dy) / len_sq
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        ex = player.x - cx
        ey = player.y - cy
        dist_sq = ex * ex + ey * ey
        min_dist = player.radius
        if dist_sq >= min_dist * min_dist or dist_sq == 0:
            return
        dist = math.sqrt(dist_sq)
        nx, ny = ex / dist, ey / dist
        player.x += nx * (min_dist - dist)
        player.y += ny * (min_dist - dist)
        vel_into = player.vx * nx + player.vy * ny
        if vel_into < 0:
            player.vx -= vel_into * nx
            player.vy -= vel_into * ny

    # ------------------------------------------------------------------ #
    #  Проверка гола                                                       #
    # ------------------------------------------------------------------ #

    def check_goal(self):
        """
        Проверяет, влетел ли мяч в ворота.

        Returns:
            'team1'  — team1 забила гол (мяч вышел за правую стену)
            'team2'  — team2 забила гол (мяч вышел за левую стену)
            None     — гола нет
        """
        ball = self.balls[0]
        gmin, gmax = self.goal_y_min, self.goal_y_max

        if ball.x + ball.radius < 0 and gmin <= ball.y <= gmax:
            self.score_team2 += 1
            return 'team2'

        if ball.x - ball.radius > self.width and gmin <= ball.y <= gmax:
            self.score_team1 += 1
            return 'team1'

        return None

    # ------------------------------------------------------------------ #
    #  Прочее                                                              #
    # ------------------------------------------------------------------ #

    def get_state(self):
        return {
            "balls":         [{"x": b.x, "y": b.y, "vx": b.vx, "vy": b.vy} for b in self.balls],
            "players_team1": [{"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy, "kick": p.is_kicking} for p in self.players_team1],
            "players_team2": [{"x": p.x, "y": p.y, "vx": p.vx, "vy": p.vy, "kick": p.is_kicking} for p in self.players_team2],
            "score_team1":   self.score_team1,
            "score_team2":   self.score_team2,
        }
