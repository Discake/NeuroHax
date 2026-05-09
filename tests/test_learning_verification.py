"""
Тесты, проверяющие, что веса модели меняются во время обучения в train_simple_model_workers.py
Эти тесты НЕ исправляют код, только проверяют обучение.
"""
import pytest
import torch
import sys
import os
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Training.PPO import PPO
from AI.Training.Memory import Memory
from Core.Domain.Entities.Map import Map


class TestWeightsChangeDuringTraining:
    """
    Тесты, проверяющие, что веса модели действительно меняются при обучении.
    Эти тесты выявляют проблемы в обучении, но не исправляют их.
    """

    @pytest.fixture
    def setup_model_and_data(self):
        """Создаёт модель и тестовые данные для обучения"""
        map_obj = Map()
        map_obj.load_random()
        
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(translator)
        policy_old = SimpleModel(translator)
        
        # Генерируем данные для обучения
        memory = Memory()
        for i in range(30):
            state = torch.randn(12)  # state_dim = 12
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])  # 5 действий
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + (i % 10) * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        return policy, policy_old, memory, translator

    def test_weights_change_after_single_update(self, setup_model_and_data):
        """
        Проверяет, что веса меняются после одного обновления PPO.
        Если тест падает - модель НЕ обучается.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем веса ДО обучения
        weights_before = {}
        for name, param in policy.named_parameters():
            weights_before[name] = param.clone().detach()
        
        # Обучаем модель
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем, что веса изменились
        weights_changed = False
        changed_params = []
        
        for name, param in policy.named_parameters():
            if name in weights_before:
                if not torch.allclose(param, weights_before[name], atol=1e-6):
                    weights_changed = True
                    changed_params.append(name)
        
        assert weights_changed, (
            "Веса модели НЕ изменились после обучения! "
            f"Параметры, которые должны были измениться: {list(policy.named_parameters())}"
        )
        print(f"✓ Веса изменились в параметрах: {changed_params}")

    def test_all_layers_receive_gradients(self, setup_model_and_data):
        """
        Проверяет, что градиенты вычисляются для всех слоёв.
        Если тест падает - некоторые слои не обучаются.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем градиенты
        layers_with_grads = []
        layers_without_grads = []
        
        for name, param in policy.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                layers_with_grads.append(name)
            else:
                layers_without_grads.append(name)
        
        # В идеале все слои должны иметь градиенты
        # Но это warning, а не error - некоторые слои могут не получить градиенты
        # в зависимости от батча
        print(f"✓ Слои с градиентами: {layers_with_grads}")
        if layers_without_grads:
            print(f"⚠ Слои без градиентов: {layers_without_grads}")
        
        # Минимум половина слоёв должны иметь градиенты
        assert len(layers_with_grads) >= len(list(policy.named_parameters())) // 2, (
            f"Слишком мало слоёв с градиентами: {len(layers_with_grads)} из {len(list(policy.named_parameters()))}"
        )

    def test_weight_change_magnitude(self, setup_model_and_data):
        """
        Проверяет, что изменение весов значительное (не численный шум).
        Если тест падает - обучение слишком медленное или есть проблемы.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем веса ДО
        weights_before = {}
        for name, param in policy.named_parameters():
            weights_before[name] = param.clone().detach()
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Вычисляем величину изменений
        total_change = 0.0
        num_params = 0
        
        for name, param in policy.named_parameters():
            if name in weights_before:
                change = (param - weights_before[name]).abs().mean().item()
                total_change += change
                num_params += 1
        
        avg_change = total_change / num_params if num_params > 0 else 0
        
        # Изменение должно быть больше численного шума
        assert avg_change > 1e-5, (
            f"Среднее изменение весов слишком маленькое: {avg_change:.8f}. "
            "Возможно, learning rate слишком низкий или градиенты не проходят."
        )
        print(f"✓ Среднее изменение весов: {avg_change:.6f}")

    def test_multiple_updates_accumulate(self, setup_model_and_data):
        """
        Проверяет, что многократные обновления накапливают изменения.
        Если тест падает - обновления могут перезаписываться или откатываться.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем начальные веса
        initial_weights = {}
        for name, param in policy.named_parameters():
            initial_weights[name] = param.clone().detach()
        
        # Делаем 3 обновления
        for ep in range(3):
            ppo.update_combined(memory, ep=ep, minibatch_size=10)
        
        # Проверяем накопленные изменения
        total_change = 0.0
        num_params = 0
        
        for name, param in policy.named_parameters():
            if name in initial_weights:
                change = (param - initial_weights[name]).abs().mean().item()
                total_change += change
                num_params += 1
        
        avg_change = total_change / num_params if num_params > 0 else 0
        
        # После 3 обновлений изменение должно быть заметным
        assert avg_change > 1e-4, (
            f"После 3 обновлений изменение весов слишком маленькое: {avg_change:.8f}"
        )
        print(f"✓ Среднее изменение после 3 обновлений: {avg_change:.6f}")

    def test_policy_old_sync(self, setup_model_and_data):
        """
        Проверяет, что policy_old синхронизируется с policy после обновления.
        Если тест падает - рассинхронизация между политиками.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # До обновления policy_old должна совпадать с policy
        weights_match_before = all(
            torch.allclose(p_param, po_param)
            for p_param, po_param in zip(policy.parameters(), policy_old.parameters())
        )
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # После обновления в PPO policy_old должна обновиться (load_state_dict)
        # Проверяем это
        weights_match_after = all(
            torch.allclose(p_param, po_param)
            for p_param, po_param in zip(policy.parameters(), policy_old.parameters())
        )
        
        # В PPO update_combined policy_old обновляется в конце
        assert weights_match_after, (
            "policy_old не синхронизирована с policy после обновления. "
            "Это может вызвать проблемы с обучением."
        )
        print("✓ policy_old синхронизирована с policy")

    def test_value_head_trains(self, setup_model_and_data):
        """
        Специфический тест для value_head - должен обучаться для PPO.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем веса value_head
        value_weight_before = policy.value_head.weight.clone().detach()
        value_bias_before = policy.value_head.bias.clone().detach()
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем изменения
        weight_changed = not torch.allclose(
            policy.value_head.weight, value_weight_before, atol=1e-6
        )
        bias_changed = not torch.allclose(
            policy.value_head.bias, value_bias_before, atol=1e-6
        )
        
        assert weight_changed or bias_changed, (
            "value_head не обучается! PPO не сможет правильно оценивать состояния."
        )
        print("✓ value_head обучается")

    def test_velocity_head_trains(self, setup_model_and_data):
        """
        Специфический тест для velocity_head - должен обучаться.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем веса velocity_head
        velocity_weight_before = policy.velocity_head.weight.clone().detach()
        velocity_bias_before = policy.velocity_head.bias.clone().detach()
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем изменения
        weight_changed = not torch.allclose(
            policy.velocity_head.weight, velocity_weight_before, atol=1e-6
        )
        bias_changed = not torch.allclose(
            policy.velocity_head.bias, velocity_bias_before, atol=1e-6
        )
        
        assert weight_changed or bias_changed, (
            "velocity_head не обучается! Модель не сможет выбирать движения."
        )
        print("✓ velocity_head обучается")

    def test_kick_head_trains(self, setup_model_and_data):
        """
        Специфический тест для kick_head - должен обучаться.
        """
        policy, policy_old, memory, translator = setup_model_and_data
        
        ppo = PPO(policy, policy_old)
        
        # Сохраняем веса kick_head
        kick_weight_before = policy.kick_head.weight.clone().detach()
        kick_bias_before = policy.kick_head.bias.clone().detach()
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем изменения
        weight_changed = not torch.allclose(
            policy.kick_head.weight, kick_weight_before, atol=1e-6
        )
        bias_changed = not torch.allclose(
            policy.kick_head.bias, kick_bias_before, atol=1e-6
        )
        
        assert weight_changed or bias_changed, (
            "kick_head не обучается! Модель не сможет выбирать удары."
        )
        print("✓ kick_head обучается")


class TestLearningSignal:
    """
    Тесты, проверяющие, что сигнал обучения (градиенты, loss) существует.
    """

    @pytest.fixture
    def setup_training(self):
        """Создаёт полную setup для обучения"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy = SimpleModel(translator1)
        policy_old = SimpleModel(translator2)
        ppo = PPO(policy, policy_old)
        
        memory = Memory()
        for i in range(40):
            state = torch.randn(12)
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + (i % 10) * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        return ppo, policy, memory

    def test_loss_is_computed(self, setup_training):
        """
        Проверяет, что loss вычисляется (не None и не NaN).
        """
        ppo, policy, memory = setup_training
        
        # Сохраняем текущие веса
        initial_weights = {k: v.clone() for k, v in policy.named_parameters()}
        
        # Обучаем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем, что веса изменились (косвенная проверка loss)
        for name, param in policy.named_parameters():
            if name in initial_weights:
                change = (param - initial_weights[name]).abs().sum().item()
                assert change > 0, f"Параметр {name} не изменился - loss не вычисляется"
                assert not torch.isnan(param).any(), f"NaN в параметре {name}"
        
        print("✓ Loss вычисляется корректно")

    def test_rewards_affect_learning(self, setup_training):
        """
        Проверяет, что разные награды приводят к разным изменениям весов.
        """
        ppo, policy, memory = setup_training
        
        # Сохраняем веса
        weights_before = {k: v.clone() for k, v in policy.named_parameters()}
        
        # Обучаем с текущими наградами
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        weights_after_high_reward = {k: v.clone() for k, v in policy.named_parameters()}
        
        # Создаём память с низкими наградами
        memory_low = Memory()
        for i in range(40):
            state = torch.randn(12)
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(-0.5)  # Низкая награда
            done = 0
            is_truncated = 0
            memory_low.store(state, action, log_prob, reward, done, is_truncated)
        
        # Сбрасываем модель к исходным весам
        for name, param in policy.named_parameters():
            param.data.copy_(weights_before[name])
        
        # Обучаем с низкими наградами
        ppo.update_combined(memory_low, ep=0, minibatch_size=10)
        
        weights_after_low_reward = {k: v.clone() for k, v in policy.named_parameters()}
        
        # Изменения должны быть разными
        diff_high = sum(
            (weights_after_high_reward[k] - weights_before[k]).abs().sum().item()
            for k in weights_before
        )
        diff_low = sum(
            (weights_after_low_reward[k] - weights_before[k]).abs().sum().item()
            for k in weights_before
        )
        
        # Разница в изменениях должна быть заметной
        relative_diff = abs(diff_high - diff_low) / max(diff_high, diff_low, 1e-8)
        
        assert relative_diff > 0.1, (
            f"Награды не влияют на обучение! relative_diff = {relative_diff:.4f}"
        )
        print(f"✓ Награды влияют на обучение (relative_diff = {relative_diff:.4f})")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
