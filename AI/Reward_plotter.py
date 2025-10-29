import matplotlib.pyplot as plt
from collections import deque

class Reward_plotter:
    def __init__(self, window_size=100):
        """
        window_size: размер окна для скользящего среднего
        """
        self.episodes = []
        self.rewards = []
        self.avg_rewards = deque(maxlen=window_size)
        self.avg_rewards_plot = []
        
        # Настройка интерактивного режима
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line1, = self.ax.plot([], [], 'b-', alpha=0.3, label='Reward per episode')
        self.line2, = self.ax.plot([], [], 'r-', linewidth=2, label=f'Moving avg ({window_size})')
        
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
    def update(self, episode, reward):
        """
        Обновить график новыми данными
        episode: номер эпизода
        reward: суммарная награда за эпизод
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.avg_rewards.append(reward)
        self.avg_rewards_plot.append(sum(self.avg_rewards) / len(self.avg_rewards))
        
        # Обновление данных линий
        self.line1.set_data(self.episodes, self.rewards)
        self.line2.set_data(self.episodes, self.avg_rewards_plot)
        
        # Автоматическое масштабирование осей
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Перерисовка
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)  # Небольшая пауза для обновления
    
    def close(self):
        """Закрыть график"""
        plt.ioff()
        plt.close()
    
    def save(self, filename='training_progress.png'):
        """Сохранить график в файл"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"График сохранён в {filename}")


# # Пример использования в цикле обучения:
# if __name__ == "__main__":
#     plotter = Reward_plotter(window_size=100)
    
#     # Симуляция обучения
#     for episode in range(1000):
#         # Здесь твой код обучения...
#         # episode_reward = train_one_episode()
        
#         # Для демонстрации генерируем случайную награду
#         import random
#         episode_reward = random.uniform(-10, 10) + episode * 0.01
        
#         # Обновление графика
#         plotter.update(episode, episode_reward)
        
#         # Печать прогресса
#         if (episode + 1) % 100 == 0:
#             avg = sum(list(plotter.avg_rewards)) / len(plotter.avg_rewards)
#             print(f"Episode {episode + 1}, Avg Reward: {avg:.2f}")
    
#     # Сохранение финального графика
#     plotter.save('final_training_progress.png')
    
#     # Показать график до закрытия окна
#     plt.ioff()
#     plt.show()
