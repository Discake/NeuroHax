"""
Живые графики статистики тренировки.
Обновляются каждые N эпизодов и сохраняются в PNG.
"""

import matplotlib
matplotlib.use('Agg')   # без GUI — рендер в файл, безопасно с multiprocessing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


class TrainingPlotter:
    def __init__(self, save_dir="models", update_every=5):
        self.save_dir = save_dir
        self.update_every = update_every
        self.save_path = os.path.join(save_dir, "training_stats.png")

        # История метрик
        self.episodes       = []
        self.goals_1        = []
        self.goals_2        = []
        self.kicks_1        = []
        self.kicks_2        = []
        self.avg_rewards    = []
        self.total_rewards  = []
        self.samples        = []

        # Скользящее среднее (окно 20)
        self._window = 20

        fig = plt.figure(figsize=(14, 9))
        fig.patch.set_facecolor('#1a1a2e')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        ax_kw = dict(facecolor='#16213e')
        self.ax_goals    = fig.add_subplot(gs[0, :2], **ax_kw)  # широкий верхний
        self.ax_balance  = fig.add_subplot(gs[0, 2],  **ax_kw)
        self.ax_kicks    = fig.add_subplot(gs[1, 0],  **ax_kw)
        self.ax_reward   = fig.add_subplot(gs[1, 1],  **ax_kw)
        self.ax_total    = fig.add_subplot(gs[1, 2],  **ax_kw)

        self.fig = fig
        self._style_axes()  # style all axes
        self.fig.suptitle("NeuroHax — Training Stats", color='white',
                          fontsize=14, fontweight='bold', y=0.98)

    # ------------------------------------------------------------------ #

    def _style_axes(self):
        for ax in [self.ax_goals, self.ax_balance, self.ax_kicks,
                   self.ax_reward, self.ax_total]:
            ax.tick_params(colors='#aaaaaa', labelsize=7)
            ax.xaxis.label.set_color('#aaaaaa')
            ax.yaxis.label.set_color('#aaaaaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

    def _rolling(self, data):
        """Скользящее среднее с окном self._window. Возвращает (x, y) одинаковой длины."""
        arr = np.array(data)
        if len(arr) < 2:
            return np.array([]), np.array([])
        k = min(self._window, len(arr))
        smooth = np.convolve(arr, np.ones(k) / k, mode='valid')  # длина: len(arr)-k+1
        ep = np.array(self.episodes)
        x = ep[k - 1:]   # всегда совпадает с длиной smooth
        return x, smooth

    # ------------------------------------------------------------------ #

    def update(self, episode, goals_1, goals_2, kicks_1, kicks_2,
               avg_reward, total_reward, n_samples):
        self.episodes.append(episode)
        self.goals_1.append(goals_1)
        self.goals_2.append(goals_2)
        self.kicks_1.append(kicks_1)
        self.kicks_2.append(kicks_2)
        self.avg_rewards.append(avg_reward)
        self.total_rewards.append(total_reward)
        self.samples.append(n_samples)

        if len(self.episodes) % self.update_every == 0 or episode == 0:
            self._draw()

    def _draw(self):
        ep = np.array(self.episodes)
        g1 = np.array(self.goals_1)
        g2 = np.array(self.goals_2)

        # ── 1. Голы ───────────────────────────────────────────────────
        ax = self.ax_goals
        ax.clear()
        ax.set_facecolor('#16213e')
        ax.fill_between(ep, g1, alpha=0.25, color='#e94560', label='Team 1')
        ax.fill_between(ep, g2, alpha=0.25, color='#0f3460', label='Team 2')
        ax.plot(ep, g1, color='#e94560', linewidth=1)
        ax.plot(ep, g2, color='#4a90e2', linewidth=1)
        rx, ry1 = self._rolling(g1)
        _,  ry2 = self._rolling(g2)
        if len(rx) > 0:
            ax.plot(rx, ry1, color='#ff6b6b', linewidth=2, label=f'T1 avg{self._window}')
            ax.plot(rx, ry2, color='#74b9ff', linewidth=2, label=f'T2 avg{self._window}')
        ax.set_title('Goals per episode', color='white', fontsize=9)
        ax.set_xlabel('Episode', fontsize=7)
        ax.legend(fontsize=6, facecolor='#1a1a2e', labelcolor='white',
                  loc='upper left', framealpha=0.7)
        self._style_ax(ax)

        # ── 2. Баланс команд ──────────────────────────────────────────
        ax = self.ax_balance
        ax.clear()
        ax.set_facecolor('#16213e')
        total_g = g1 + g2
        with np.errstate(divide='ignore', invalid='ignore'):
            share1 = np.where(total_g > 0, g1 / total_g * 100, 50.0)
        ax.plot(ep, share1, color='#e94560', linewidth=1.5)
        ax.axhline(50, color='#aaaaaa', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.fill_between(ep, share1, 50,
                        where=share1 >= 50, alpha=0.2, color='#e94560')
        ax.fill_between(ep, share1, 50,
                        where=share1 < 50,  alpha=0.2, color='#4a90e2')
        ax.set_ylim(0, 100)
        ax.set_title('Team 1 goal share (%)', color='white', fontsize=9)
        ax.set_xlabel('Episode', fontsize=7)
        self._style_ax(ax)

        # ── 3. Kicks per episode ──────────────────────────────────────
        ax = self.ax_kicks
        ax.clear()
        ax.set_facecolor('#16213e')
        k1 = np.array(self.kicks_1)
        k2 = np.array(self.kicks_2)
        ax.fill_between(ep, k1, alpha=0.25, color='#e94560')
        ax.fill_between(ep, k2, alpha=0.25, color='#4a90e2')
        ax.plot(ep, k1, color='#e94560', linewidth=1, label='T1')
        ax.plot(ep, k2, color='#4a90e2', linewidth=1, label='T2')
        rx, rk1 = self._rolling(k1)
        _,  rk2 = self._rolling(k2)
        if len(rx) > 0:
            ax.plot(rx, rk1, color='#ff6b6b', linewidth=2)
            ax.plot(rx, rk2, color='#74b9ff', linewidth=2)
        ax.set_title('Kicks per episode', color='white', fontsize=9)
        ax.set_xlabel('Episode', fontsize=7)
        ax.legend(fontsize=6, facecolor='#1a1a2e', labelcolor='white',
                  loc='upper left', framealpha=0.7)
        self._style_ax(ax)

        # ── 4. Avg reward per step ────────────────────────────────────
        ax = self.ax_reward
        ax.clear()
        ax.set_facecolor('#16213e')
        ar = np.array(self.avg_rewards)
        ax.plot(ep, ar, color='#a29bfe', linewidth=1, alpha=0.5)
        rx, ry = self._rolling(ar)
        if len(rx) > 0:
            ax.plot(rx, ry, color='#a29bfe', linewidth=2)
        ax.set_title('Avg reward / step', color='white', fontsize=9)
        ax.set_xlabel('Episode', fontsize=7)
        self._style_ax(ax)

        # ── 5. Total reward ───────────────────────────────────────────
        ax = self.ax_total
        ax.clear()
        ax.set_facecolor('#16213e')
        tr = np.array(self.total_rewards)
        ax.plot(ep, tr, color='#55efc4', linewidth=1, alpha=0.5)
        rx, ry = self._rolling(tr)
        if len(rx) > 0:
            ax.plot(rx, ry, color='#55efc4', linewidth=2)
        ax.set_title('Total reward (all workers)', color='white', fontsize=9)
        ax.set_xlabel('Episode', fontsize=7)
        self._style_ax(ax)

        # Итоговая статистика в заголовке
        last_ep   = self.episodes[-1]
        k1        = np.array(self.kicks_1)
        k2        = np.array(self.kicks_2)
        total_g   = g1 + g2
        best_goals = int(max(total_g)) if len(total_g) > 0 else 0
        avg20_goals = np.mean(total_g[-self._window:]) if len(total_g) > 0 else 0
        # Конверсия: доля ударов, завершившихся голом (avg по последним 20 эпизодам)
        total_k = k1 + k2
        with np.errstate(divide='ignore', invalid='ignore'):
            conv = np.where(total_k > 0, total_g / total_k * 100, 0.0)
        avg20_conv = np.mean(conv[-self._window:]) if len(conv) > 0 else 0
        self.fig.suptitle(
            f"NeuroHax — Training Stats    "
            f"ep {last_ep}  |  best goals/ep: {best_goals}  |  "
            f"avg20 goals: {avg20_goals:.1f}  |  "
            f"kick→goal conv: {avg20_conv:.1f}%",
            color='white', fontsize=11, fontweight='bold', y=0.99
        )

        self.fig.savefig(self.save_path, dpi=110,
                         bbox_inches='tight', facecolor='#1a1a2e')

    @staticmethod
    def _style_ax(ax):
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')

    def close(self):
        self._draw()   # финальный рендер
        plt.close(self.fig)
        print(f"  -> График сохранён: {self.save_path}")
