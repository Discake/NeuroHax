"""
Тесты симметрии SimpleModelTranslator.

Инвариант: если отразить всё поле по X (x → FIELD_W - x, vx → -vx),
то team1 должна видеть то же состояние, что team2 в оригинале.
Нарушение → агент не понимает, в какие ворота бить.

Индексы тензора состояния:
  [0-3]   my_pos_x, my_pos_y, my_vel_x, my_vel_y
  [4-5]   rel_ball_pos_x, rel_ball_pos_y
  [6-7]   ball_vel_x, ball_vel_y
  [8-11]  rel_opp_pos_x, rel_opp_pos_y, opp_vel_x, opp_vel_y
  [12]    vel_toward_ball
  [13-14] ball_to_opp_goal_x, ball_to_opp_goal_y
  [15-16] ball_pos_x, ball_pos_y  (абсолютные, после flip)
  [17-18] opp_pos_x, opp_pos_y   (абсолютные, после flip)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
import Constants

FIELD_W = Constants.field_size[0]   # 600
FIELD_H = Constants.field_size[1]   # 420
CX = Constants.x_center             # 300  (центр поля по X)
CY = Constants.y_center             # 210  (центр поля по Y)
TOL = 1e-4


# ── Вспомогательные классы ────────────────────────────────────────────────────

class FakeEntity:
    """Имитирует Ball / Player (совместим с реальным интерфейсом)."""
    def __init__(self, x, y, vx=0.0, vy=0.0):
        self.x, self.y   = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.radius      = 20
        self.is_kicking  = False

    @property
    def position(self):
        return [self.x, self.y]

    @property
    def velocity(self):
        return [self.vx, self.vy]


class FakeMap:
    def __init__(self, p1x, p1y, p2x, p2y, bx, by,
                 p1vx=0.0, p1vy=0.0, p2vx=0.0, p2vy=0.0,
                 bvx=0.0, bvy=0.0):
        self.players_team1 = [FakeEntity(p1x, p1y, p1vx, p1vy)]
        self.players_team2 = [FakeEntity(p2x, p2y, p2vx, p2vy)]
        ball = FakeEntity(bx, by, bvx, bvy)
        ball.radius = 10
        self.balls = [ball]


def translators(m: FakeMap):
    t1 = SimpleModelTranslator(m, m.players_team1[0], is_team_1=True)
    t2 = SimpleModelTranslator(m, m.players_team2[0], is_team_1=False)
    return t1.translate({}), t2.translate({})


def mirror_swap(m: FakeMap) -> FakeMap:
    """
    Горизонтальное отражение + смена ролей команд.
    Новая team1 = зеркало старой team2, новая team2 = зеркало старой team1.
    Инвариант: team1.translate(m) == team2.translate(mirror_swap(m))
    """
    p1, p2, b = m.players_team1[0], m.players_team2[0], m.balls[0]
    return FakeMap(
        FIELD_W - p2.x, p2.y,   # new team1 ← mirror(old team2)
        FIELD_W - p1.x, p1.y,   # new team2 ← mirror(old team1)
        FIELD_W - b.x,  b.y,
        -p2.vx, p2.vy,  -p1.vx, p1.vy,  -b.vx, b.vy,
    )


# ── Фикстуры ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sym_map():
    """Идеально симметричная карта: игроки и мяч зеркально расположены."""
    # p1 at (150, 210), p2 at (450, 210) — зеркально относительно центра (300)
    # мяч в центре поля
    return FakeMap(
        p1x=150, p1y=CY,  p2x=FIELD_W - 150, p2y=CY,
        bx=CX,   by=CY,
        p1vx=50, p2vx=-50,   # зеркальные скорости
    )


@pytest.fixture
def ball_left_map():
    """Мяч в левой четверти поля — ближе к воротам team2."""
    return FakeMap(
        p1x=100, p1y=CY,  p2x=500, p2y=CY,
        bx=150,  by=CY,
    )


@pytest.fixture
def ball_right_map():
    """Мяч в правой четверти поля — ближе к воротам team1."""
    return FakeMap(
        p1x=100, p1y=CY,  p2x=500, p2y=CY,
        bx=450,  by=CY,
    )


# ── Тесты ────────────────────────────────────────────────────────────────────

class TestFullSymmetry:
    """В идеально симметричной ситуации оба игрока обязаны видеть одинаковое состояние."""

    def test_my_position_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[0] - s2[0]) < TOL, (
            f"my_pos_x: team1={s1[0]:.4f}, team2={s2[0]:.4f} — нарушена симметрия позиции")
        assert abs(s1[1] - s2[1]) < TOL, (
            f"my_pos_y: team1={s1[1]:.4f}, team2={s2[1]:.4f}")

    def test_my_velocity_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[2] - s2[2]) < TOL, (
            f"my_vel_x: team1={s1[2]:.4f}, team2={s2[2]:.4f} — flip скорости не работает")
        assert abs(s1[3] - s2[3]) < TOL, (
            f"my_vel_y: team1={s1[3]:.4f}, team2={s2[3]:.4f}")

    def test_rel_ball_pos_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[4] - s2[4]) < TOL, (
            f"rel_ball_pos_x: team1={s1[4]:.4f}, team2={s2[4]:.4f}")
        assert abs(s1[5] - s2[5]) < TOL, (
            f"rel_ball_pos_y: team1={s1[5]:.4f}, team2={s2[5]:.4f}")

    def test_ball_velocity_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[6] - s2[6]) < TOL, (
            f"ball_vel_x: team1={s1[6]:.4f}, team2={s2[6]:.4f}")
        assert abs(s1[7] - s2[7]) < TOL, (
            f"ball_vel_y: team1={s1[7]:.4f}, team2={s2[7]:.4f}")

    def test_opponent_pos_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[8]  - s2[8])  < TOL, (
            f"rel_opp_pos_x: team1={s1[8]:.4f}, team2={s2[8]:.4f}")
        assert abs(s1[9]  - s2[9])  < TOL, (
            f"rel_opp_pos_y: team1={s1[9]:.4f}, team2={s2[9]:.4f}")
        assert abs(s1[10] - s2[10]) < TOL, (
            f"opp_vel_x: team1={s1[10]:.4f}, team2={s2[10]:.4f}")

    def test_vel_toward_ball_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[12] - s2[12]) < TOL, (
            f"vel_toward_ball: team1={s1[12]:.4f}, team2={s2[12]:.4f}")

    def test_ball_to_opp_goal_symmetric(self, sym_map):
        """
        Ключевой тест: в симметричной ситуации оба игрока должны видеть
        одинаковое расстояние до ворот соперника.

        ИЗВЕСТНЫЙ БАГ: переводчик использует window_size[0]=840 вместо
        field_size[0]=600 для правых ворот (team1), что даёт
        (840-300)/420=1.286 у team1 против 300/420=0.714 у team2.
        """
        s1, s2 = translators(sym_map)
        assert abs(s1[13] - s2[13]) < TOL, (
            f"ball_to_opp_goal_x: team1={s1[13]:.4f}, team2={s2[13]:.4f}\n"
            f"  Подсказка: проверь opp_goal_x_world в SimpleModelTranslator — "
            f"должно быть Constants.field_size[0]={FIELD_W}, не window_size[0]={Constants.window_size[0]}")
        assert abs(s1[14] - s2[14]) < TOL, (
            f"ball_to_opp_goal_y: team1={s1[14]:.4f}, team2={s2[14]:.4f}")

    def test_absolute_ball_pos_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        # Мяч в центре → ball_pos_x = 0 для обоих после flip
        assert abs(s1[15] - s2[15]) < TOL, (
            f"ball_pos_x (abs): team1={s1[15]:.4f}, team2={s2[15]:.4f}")
        assert abs(s1[16] - s2[16]) < TOL, (
            f"ball_pos_y (abs): team1={s1[16]:.4f}, team2={s2[16]:.4f}")

    def test_absolute_opp_pos_symmetric(self, sym_map):
        s1, s2 = translators(sym_map)
        assert abs(s1[17] - s2[17]) < TOL, (
            f"opp_pos_x (abs): team1={s1[17]:.4f}, team2={s2[17]:.4f}")
        assert abs(s1[18] - s2[18]) < TOL, (
            f"opp_pos_y (abs): team1={s1[18]:.4f}, team2={s2[18]:.4f}")


class TestGoalDirection:
    """Вектор ball_to_opp_goal должен осмысленно указывать в сторону ворот соперника."""

    def test_ball_at_center_goal_direction_positive(self, sym_map):
        """Мяч в центре → вектор к воротам соперника указывает вправо (>0)."""
        s1, s2 = translators(sym_map)
        assert s1[13] > 0, f"team1: ball_to_opp_goal_x должен быть >0, но = {s1[13]:.4f}"
        assert s2[13] > 0, f"team2: ball_to_opp_goal_x должен быть >0, но = {s2[13]:.4f}"

    def test_ball_near_opponents_goal_smaller_distance(self):
        """Чем ближе мяч к воротам соперника, тем меньше ball_to_opp_goal_x."""
        # team1: ворота соперника справа (x=FIELD_W)
        # мяч на x=500 ближе к воротам, чем на x=300
        m_far  = FakeMap(100, CY, 500, CY, 300, CY)   # мяч далеко от правых ворот
        m_near = FakeMap(100, CY, 500, CY, 500, CY)   # мяч близко к правым воротам

        t1_far  = SimpleModelTranslator(m_far,  m_far.players_team1[0],  is_team_1=True)
        t1_near = SimpleModelTranslator(m_near, m_near.players_team1[0], is_team_1=True)

        s_far  = t1_far.translate({})
        s_near = t1_near.translate({})

        assert s_near[13] < s_far[13], (
            f"Мяч у ворот должен давать меньший ball_to_opp_goal_x: "
            f"far={s_far[13]:.4f}, near={s_near[13]:.4f}")

    def test_ball_moving_toward_goal_positive_vel(self):
        """Мяч, летящий в сторону ворот соперника, должен давать положительный ball_vel_x."""
        # team1 атакует вправо → ball.vx > 0 должно давать ball_vel_x > 0
        m = FakeMap(100, CY, 500, CY, CX, CY, bvx=100)
        s1, s2 = translators(m)
        assert s1[6] > 0, f"team1 ball_vel_x при vx>0 должен быть >0: {s1[6]:.4f}"
        assert s2[6] < 0, f"team2 ball_vel_x при vx>0 должен быть <0: {s2[6]:.4f}"

    def test_ball_at_opponent_goal_line(self):
        """Когда мяч на линии ворот соперника, ball_to_opp_goal_x должен быть ~0."""
        # team1: правые ворота на x=FIELD_W=600
        m = FakeMap(100, CY, 500, CY, FIELD_W, CY)
        t1 = SimpleModelTranslator(m, m.players_team1[0], is_team_1=True)
        s1 = t1.translate({})
        assert abs(s1[13]) < TOL, (
            f"Мяч на линии ворот: ball_to_opp_goal_x должен быть ~0, но = {s1[13]:.4f}\n"
            f"  Подсказка: цель правых ворот задана как {Constants.window_size[0]} "
            f"вместо {FIELD_W}")

    def test_ball_at_own_goal_line(self):
        """Когда мяч на линии своих ворот (x=0), team2 должна видеть ~0 distance."""
        m = FakeMap(100, CY, 500, CY, 0.0, CY)
        t2 = SimpleModelTranslator(m, m.players_team2[0], is_team_1=False)
        s2 = t2.translate({})
        # team2 атакует влево → ворота соперника на x=0
        # ball_to_opp_goal_x ≈ 0
        assert abs(s2[13]) < TOL, (
            f"team2 мяч у x=0: ball_to_opp_goal_x должен быть ~0, но = {s2[13]:.4f}")


class TestMirrorInvariant:
    """
    Строгий инвариант: team1.translate(m) == team2.translate(mirror_swap(m)).

    Смысл: если зеркально отразить игру и поменять роли команд, то
    team2 должна видеть ровно то же состояние, что team1 видела в оригинале.
    Это гарантирует, что единственная модель корректно управляет обоими игроками.
    """

    def _check_mirror(self, m: FakeMap):
        m_ms = mirror_swap(m)
        t1_orig = SimpleModelTranslator(m,    m.players_team1[0],    is_team_1=True)
        t2_ms   = SimpleModelTranslator(m_ms, m_ms.players_team2[0], is_team_1=False)
        s1 = t1_orig.translate({})
        s2 = t2_ms.translate({})
        for i in range(len(s1)):
            assert abs(s1[i] - s2[i]) < TOL, (
                f"Инвариант нарушен на индексе {i}: "
                f"team1(orig)={s1[i]:.4f}, team2(mirror_swap)={s2[i]:.4f}")

    def test_mirror_invariant_center_ball(self):
        """Мяч в центре, игроки симметрично."""
        self._check_mirror(FakeMap(150, CY, 450, CY, CX, CY, p1vx=30, p2vx=-30))

    def test_mirror_invariant_ball_left(self):
        """Мяч в левой трети поля."""
        self._check_mirror(FakeMap(100, CY, 500, CY, 150, CY))

    def test_mirror_invariant_ball_right(self):
        """Мяч в правой трети поля."""
        self._check_mirror(FakeMap(100, CY, 500, CY, 450, CY))

    def test_mirror_invariant_ball_off_center_y(self):
        """Мяч не по центру по Y."""
        self._check_mirror(FakeMap(200, 100, 400, 320, CX, 80, p1vx=20, p2vx=-20, bvx=50, bvy=-30))

    def test_mirror_invariant_with_velocities(self):
        """Различные скорости игроков и мяча."""
        self._check_mirror(FakeMap(
            200, 150,  400, 270,  CX, 100,
            p1vx=80, p1vy=30,  p2vx=-80, p2vy=30,
            bvx=120, bvy=-40,
        ))


class TestCanonicalForm:
    """Проверяет корректность канонического представления."""

    def test_self_goal_is_left_own_goal_is_right_for_both(self):
        """
        В канонической системе «свои» ворота всегда слева (my_pos_x < 0
        если игрок на своей половине), «чужие» всегда справа.
        Проверяем через ball_to_opp_goal_x > 0 при мяче в центре.
        """
        m = FakeMap(150, CY, 450, CY, CX, CY)
        s1, s2 = translators(m)
        assert s1[13] > 0, "team1: ворота соперника должны быть 'справа' (ball_to_opp_goal_x > 0)"
        assert s2[13] > 0, "team2: ворота соперника должны быть 'справа' (ball_to_opp_goal_x > 0)"

    def test_player_on_own_half_has_negative_my_pos_x(self):
        """Игрок на своей половине должен иметь отрицательный my_pos_x."""
        # team1 защищает левые ворота → своя половина x < CX=300
        m = FakeMap(100, CY, 500, CY, CX, CY)
        s1, s2 = translators(m)
        assert s1[0] < 0, f"team1 на своей половине: my_pos_x должен быть <0, но = {s1[0]:.4f}"
        assert s2[0] < 0, f"team2 на своей половине: my_pos_x должен быть <0, но = {s2[0]:.4f}"

    def test_player_on_opp_half_has_positive_my_pos_x(self):
        """Игрок на чужой половине должен иметь положительный my_pos_x."""
        m = FakeMap(500, CY, 100, CY, CX, CY)
        s1, s2 = translators(m)
        assert s1[0] > 0, f"team1 на чужой половине: my_pos_x={s1[0]:.4f}"
        assert s2[0] > 0, f"team2 на чужой половине: my_pos_x={s2[0]:.4f}"

    def test_opponent_is_always_canonical_right_when_behind_ball(self):
        """Если соперник дальше от мяча по X, rel_opp_pos_x > rel_ball_pos_x."""
        # team1 рядом с мячом, team2 далеко справа
        m = FakeMap(280, CY, 580, CY, CX, CY)
        s1, _ = translators(m)
        # rel_ball_pos_x = мяч относительно меня (мяч правее → >0)
        # rel_opp_pos_x  = соперник относительно меня (ещё правее → больше)
        assert s1[8] > s1[4], (
            f"Соперник дальше мяча: rel_opp_pos_x={s1[8]:.4f} должен > rel_ball_pos_x={s1[4]:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
