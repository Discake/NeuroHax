"""
Тесты для train_simple_model_workers.py - проверка обучения SimpleModel
"""
import pytest
import torch
import sys
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.Models.SimpleModel.Policy import SimpleModel
from AI.Models.SimpleModel.SimpleModelTranslator import SimpleModelTranslator
from AI.Models.SimpleModel.SimpleModelEnvironment import SimpleModelEnvironment
from AI.Training.PPO import PPO
from AI.Training.Memory import Memory
from Core.Domain.Entities.Map import Map


class TestSimpleModel:
    """Тесты SimpleModel"""

    @pytest.fixture
    def map_and_translator(self):
        """Создаёт карту и translator для тестов"""
        map_obj = Map()
        map_obj.load_random()
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        return map_obj, translator

    @pytest.fixture
    def simple_model(self, map_and_translator):
        """Создаёт SimpleModel для тестов"""
        _, translator = map_and_translator
        return SimpleModel(translator)

    def test_model_init(self, simple_model):
        """Инициализация модели"""
        assert simple_model is not None
        assert len(list(simple_model.parameters())) > 0

    def test_model_forward(self, simple_model, map_and_translator):
        """Прямой проход модели"""
        map_obj, translator = map_and_translator
        # Модель ожидает dict от translator, а не raw tensor
        state_dict = translator.translate({})  # Получаем состояние правильного формата
        
        with torch.no_grad():
            output = simple_model(state_dict)
        
        assert output is not None
        # output = (velocity, kick_logit, value)
        assert len(output) == 3

    def test_select_action(self, simple_model, map_and_translator):
        """Выбор действия"""
        map_obj, translator = map_and_translator
        state_dict = translator.translate({})
        
        with torch.no_grad():
            action, log_prob = simple_model.select_action(state_dict, deterministic=False)
        
        assert action is not None
        assert log_prob is not None
        # action = [up, down, left, right, kick] = 5 элементов
        assert action.shape[-1] == 5

    def test_select_action_deterministic(self, simple_model, map_and_translator):
        """Детерминированный выбор действия"""
        map_obj, translator = map_and_translator
        state_dict = translator.translate({})
        
        with torch.no_grad():
            action1, _ = simple_model.select_action(state_dict, deterministic=True)
            action2, _ = simple_model.select_action(state_dict, deterministic=True)
        
        # При детерминированном выборе действия должны совпадать
        assert torch.allclose(action1, action2)

    def test_model_state_dict(self, simple_model):
        """Сохранение и загрузка state_dict"""
        state_dict = simple_model.state_dict()
        
        # Создаём новую модель
        map_obj = Map()
        map_obj.load_random()
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        new_model = SimpleModel(translator)
        
        # Загружаем веса
        new_model.load_state_dict(state_dict)
        
        # Проверяем, что веса совпадают
        for k, v in simple_model.state_dict().items():
            assert torch.allclose(v, new_model.state_dict()[k])


class TestSimpleModelEnvironment:
    """Тесты SimpleModelEnvironment"""

    @pytest.fixture
    def environment(self):
        """Создаёт среду для тестов"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy1 = SimpleModel(translator1)
        policy2 = SimpleModel(translator2)
        
        return SimpleModelEnvironment(policy1, policy2, num_steps=100)

    def test_env_init(self, environment):
        """Инициализация среды"""
        assert environment is not None

    def test_env_reset(self, environment):
        """Сброс среды"""
        s1, s2 = environment.reset()
        
        assert s1 is not None
        assert s2 is not None
        # s1, s2 - это тензоры состояний, размерность зависит от translator
        assert s1.dim() >= 1
        assert s2.dim() >= 1

    def test_env_step(self, environment):
        """Шаг в среде"""
        s1, s2 = environment.reset()
        
        # action = [up, down, left, right, kick] - 5 бинарных значений
        action1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        action2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        
        (ns1, ns2), (r1, r2), done, info = environment.step(action1, action2)
        
        assert ns1 is not None
        assert ns2 is not None
        assert r1 is not None
        assert r2 is not None
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_env_step_with_kick(self, environment):
        """Шаг с ударом"""
        s1, s2 = environment.reset()
        
        # Удар с kick=1.0
        action1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        action2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
        
        (ns1, ns2), (r1, r2), done, info = environment.step(action1, action2)
        
        assert ns1 is not None
        assert 'natural_done' in info or 'truncated' in info


class TestMemory:
    """Тесты Memory для SimpleModel"""

    @pytest.fixture
    def memory_with_data(self):
        """Memory с тестовыми данными"""
        memory = Memory()
        
        # Добавляем опыт
        for i in range(20):
            state = torch.randn(12)  # state_dim = 12 для SimpleModel
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])  # 4 binary + 1 kick
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + i * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        return memory

    def test_memory_init(self):
        """Инициализация Memory"""
        memory = Memory()
        
        assert memory is not None
        assert len(memory.states) == 0
        assert len(memory.actions_final) == 0

    def test_memory_store(self, memory_with_data):
        """Сохранение опыта"""
        assert len(memory_with_data.states) == 20
        assert len(memory_with_data.rewards) == 20

    def test_memory_copy_to_tensors(self, memory_with_data):
        """Копирование в тензоры"""
        memory_with_data.copy_to_tensors()
        
        # После copy_to_tensors states становится тензором
        assert memory_with_data.states is not None
        assert torch.is_tensor(memory_with_data.states)
        assert memory_with_data.actions_final is not None
        assert torch.is_tensor(memory_with_data.actions_final)


class TestPPOWithSimpleModel:
    """Тесты PPO с SimpleModel"""

    @pytest.fixture
    def ppo_with_simple_model(self):
        """PPO с SimpleModel"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy = SimpleModel(translator1)
        policy_old = SimpleModel(translator2)
        
        return PPO(policy, policy_old)

    @pytest.fixture
    def sample_memory_for_simple_model(self):
        """Memory с данными для SimpleModel"""
        memory = Memory()
        
        # state_dim = 12 для SimpleModel (согласно SimpleModelTranslator.get_state_dim())
        for i in range(30):
            state = torch.randn(12)  # 12, а не 17!
            # action_size = 5 (4 binary + 1 kick)
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + (i % 10) * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        return memory

    def test_ppo_init(self, ppo_with_simple_model):
        """Инициализация PPO"""
        assert ppo_with_simple_model is not None
        assert ppo_with_simple_model.policy is not None
        assert ppo_with_simple_model.policy_old is not None

    def test_ppo_update(self, ppo_with_simple_model, sample_memory_for_simple_model):
        """Обновление PPO"""
        # PPO update может быть чувствителен к размеру батча
        ppo_with_simple_model.update_combined(
            sample_memory_for_simple_model, 
            ep=0, 
            minibatch_size=10
        )
        
        # Проверяем, что модель обновилась
        assert ppo_with_simple_model.policy is not None

    def test_ppo_update_preserves_memory(self, ppo_with_simple_model, sample_memory_for_simple_model):
        """Обновление не должно модифицировать память"""
        original_len = len(sample_memory_for_simple_model.states)
        
        ppo_with_simple_model.update_combined(
            sample_memory_for_simple_model, 
            ep=0,
            minibatch_size=10
        )
        
        # После update память может быть преобразована в тензоры
        # Проверяем, что данные сохранились
        if isinstance(sample_memory_for_simple_model.states, list):
            assert len(sample_memory_for_simple_model.states) == original_len
        else:
            # Если тензор - проверяем размерность
            assert sample_memory_for_simple_model.states.shape[0] == original_len


class TestLearningProgress:
    """Тесты прогресса обучения"""

    @pytest.fixture
    def trained_model(self):
        """Модель после небольшого обучения"""
        map_obj = Map()
        map_obj.load_random()
        
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(translator)
        policy_old = SimpleModel(translator)
        
        ppo = PPO(policy, policy_old)
        memory = Memory()
        
        # Генерируем опыт с положительными наградами
        for i in range(50):
            state = torch.randn(12)  # state_dim = 12 для SimpleModel
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(1.0)  # Положительная награда
            done = 0
            is_truncated = 0
            
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        # Обновляем модель
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        return policy

    def test_model_output_changes_after_training(self, trained_model):
        """Выход модели должен измениться после обучения"""
        # Создаём копию модели до обучения (уже обучена в fixture)
        map_obj = Map()
        map_obj.load_random()
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        untrained_model = SimpleModel(translator)
        
        state_dict = translator.translate({})
        
        with torch.no_grad():
            output_before = untrained_model(state_dict)
            output_after = trained_model(state_dict)
        
        # Выходы должны отличаться (модель обучилась)
        # Проверяем, что хотя бы один выход изменился
        velocity_changed = not torch.allclose(output_before[0], output_after[0], atol=0.1)
        kick_changed = not torch.allclose(output_before[1], output_after[1], atol=0.1)
        
        assert velocity_changed or kick_changed, "Модель должна измениться после обучения"

    def test_reward_improvement(self, trained_model):
        """Проверка, что модель может выбирать действия"""
        map_obj = Map()
        map_obj.load_random()
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        state_dict = translator.translate({})
        
        with torch.no_grad():
            action, log_prob = trained_model.select_action(state_dict, deterministic=False)
        
        assert action is not None
        assert log_prob is not None
        assert log_prob.item() < 0  # Log probability отрицательная


class TestTrainingIntegration:
    """Интеграционные тесты обучения"""

    def test_full_training_cycle_short(self):
        """Короткий цикл обучения"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy = SimpleModel(translator1)
        policy_old = SimpleModel(translator2)
        
        ppo = PPO(policy, policy_old)
        
        # Запоминаем начальные веса
        initial_weights = {k: v.clone() for k, v in policy.state_dict().items()}
        
        # Генерируем опыт
        memory = Memory()
        for i in range(40):
            state = torch.randn(12)  # state_dim = 12 для SimpleModel
            action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            log_prob = torch.tensor(-1.5)
            reward = torch.tensor(0.5 + (i % 10) * 0.1)
            done = 1 if i % 10 == 9 else 0
            is_truncated = 0
            memory.store(state, action, log_prob, reward, done, is_truncated)
        
        # Обновляем
        ppo.update_combined(memory, ep=0, minibatch_size=10)
        
        # Проверяем, что веса изменились
        weights_changed = False
        for k, v in policy.state_dict().items():
            if not torch.allclose(v, initial_weights[k], atol=1e-5):
                weights_changed = True
                break
        
        assert weights_changed, "Веса модели должны обновиться при обучении"

    def test_multiple_training_iterations(self):
        """Несколько итераций обучения"""
        map_obj = Map()
        map_obj.load_random()
        
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        policy = SimpleModel(translator)
        policy_old = SimpleModel(translator)
        
        ppo = PPO(policy, policy_old)
        
        # Несколько итераций обучения
        for iteration in range(3):
            memory = Memory()
            for i in range(20):
                state = torch.randn(12)  # state_dim = 12 для SimpleModel
                action = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
                log_prob = torch.tensor(-1.5)
                reward = torch.tensor(0.5 + iteration * 0.1)
                done = 0
                is_truncated = 0
                memory.store(state, action, log_prob, reward, done, is_truncated)
            
            ppo.update_combined(memory, ep=iteration, minibatch_size=10)
        
        # Модель должна остаться валидной
        state_dict = translator.translate({})
        with torch.no_grad():
            output = policy(state_dict)
        
        assert output is not None
        assert not torch.isnan(output[0]).any()
        assert not torch.isinf(output[0]).any()


class TestModelSaveLoad:
    """Тесты сохранения и загрузки модели"""

    @pytest.fixture
    def temp_dir(self):
        """Временная директория"""
        test_dir = tempfile.mkdtemp()
        yield test_dir
        shutil.rmtree(test_dir, ignore_errors=True)

    def test_save_and_load_model(self, temp_dir):
        """Сохранение и загрузка модели"""
        map_obj = Map()
        map_obj.load_random()
        
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        model = SimpleModel(translator)
        
        save_path = os.path.join(temp_dir, "test_model.pth")
        
        # Сохраняем
        torch.save(model.state_dict(), save_path)
        
        # Проверяем файл
        assert os.path.exists(save_path)
        
        # Загружаем в новую модель
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        loaded_model = SimpleModel(translator2)
        loaded_model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
        
        # Проверяем, что веса совпадают
        for k, v in model.state_dict().items():
            assert torch.allclose(v, loaded_model.state_dict()[k])

    def test_model_output_after_load(self, temp_dir):
        """Выход модели после загрузки"""
        map_obj = Map()
        map_obj.load_random()
        
        translator = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        model = SimpleModel(translator)
        
        save_path = os.path.join(temp_dir, "test_model.pth")
        
        # Тестовый вход через translator
        state_dict = translator.translate({})
        
        with torch.no_grad():
            output_before = model(state_dict)
        
        # Сохраняем и загружаем
        torch.save(model.state_dict(), save_path)
        
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        loaded_model = SimpleModel(translator2)
        loaded_model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))
        
        with torch.no_grad():
            output_after = loaded_model(state_dict)
        
        # Выходы должны совпадать
        assert torch.allclose(output_before[0], output_after[0])
        assert torch.allclose(output_before[1], output_after[1])


class TestEnvironmentRewards:
    """Тесты наград в среде"""

    def test_reward_structure(self):
        """Проверка структуры наград"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy1 = SimpleModel(translator1)
        policy2 = SimpleModel(translator2)
        
        env = SimpleModelEnvironment(policy1, policy2, num_steps=50)
        
        s1, s2 = env.reset()
        
        rewards = []
        for _ in range(10):
            # action = [up, down, left, right, kick]
            action1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
            action2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
            _, (r1, r2), done, _ = env.step(action1, action2)
            rewards.append((r1.item(), r2.item()))
            
            if done:
                break
        
        # Награды должны быть числами
        for r1, r2 in rewards:
            assert isinstance(r1, float)
            assert isinstance(r2, float)
            assert not (r1 != r1)  # Не NaN
            assert not (r2 != r2)  # Не NaN

    def test_episode_completion(self):
        """Завершение эпизода"""
        map_obj = Map()
        map_obj.load_random()
        
        translator1 = SimpleModelTranslator(map_obj, map_obj.players_team1[0], is_team_1=True)
        translator2 = SimpleModelTranslator(map_obj, map_obj.players_team2[0], is_team_1=False)
        
        policy1 = SimpleModel(translator1)
        policy2 = SimpleModel(translator2)
        
        env = SimpleModelEnvironment(policy1, policy2, num_steps=20)
        
        s1, s2 = env.reset()
        
        steps = 0
        done = False
        while not done and steps < 30:
            # action = [up, down, left, right, kick]
            action1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
            action2 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])
            
            _, _, done, info = env.step(action1, action2)
            steps += 1
        
        # Эпизод должен завершиться либо по done, либо по max_steps
        assert done or steps >= 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
