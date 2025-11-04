# profile_simulation.py
import cProfile
import pstats
import torch
import time
from contextlib import contextmanager

from Core.Objects.Map import Map

@contextmanager
def timer(name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f}s")

def detailed_profiling():
    """Детальное профилирование отдельных частей"""
    map_obj = Map()
    
    # Профилирование движения без коллизий
    with timer("Movement only"):
        for _ in range(1000):
            for ball in map_obj.all_balls:
                ball.move(map_obj.time_increment)
    
    # Профилирование коллизий
    with timer("Collisions only"):
        for _ in range(100):
            for i in range(len(map_obj.all_balls)):
                ball = map_obj.all_balls[i]
                for j in range(i + 1, len(map_obj.all_balls)):
                    other_ball = map_obj.all_balls[j]
                    if ball.detect_collision(other_ball):
                        ball.resolve_collision(other_ball)
    
    # Профилирование применения трения
    with timer("Air resistance"):
        for _ in range(1000):
            for ball in map_obj.all_balls:
                ball.apply_air_resistance(map_obj.time_increment)

def pytorch_tensor_profiling():
    """Профилирование операций с тензорами"""
    
    # Тест векторизованных операций vs циклов
    N = 100  # количество объектов
    positions = torch.randn(N, 2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    velocities = torch.randn(N, 2, device=positions.device)
    dt = 0.01
    
    # Цикл (медленно)
    with timer("Loop-based movement"):
        for _ in range(1000):
            for i in range(N):
                positions[i] = positions[i] + velocities[i] * dt
    
    # Векторизованный (быстро)
    with timer("Vectorized movement"):
        for _ in range(1000):
            positions = positions + velocities * dt

if __name__ == "__main__":
    print("=== Детальное профилирование ===")
    detailed_profiling()
    
    print("\n=== Профилирование тензорных операций ===")
    pytorch_tensor_profiling()
    
    print("\n=== Общее профилирование с cProfile ===")
    def full_simulation():
        map_obj = Map()
        for _ in range(100):
            map_obj.move_balls()
    
    cProfile.run('full_simulation()', 'simulation_profile.prof')
    
    # Анализ результатов
    stats = pstats.Stats('simulation_profile.prof')
    print("\n=== Топ-20 функций по времени ===")
    stats.sort_stats('cumulative').print_stats(20)
    
    print("\n=== Топ-10 самых медленных функций ===")
    stats.sort_stats('tottime').print_stats(10)
