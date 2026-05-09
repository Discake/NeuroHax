# Тесты для проекта NeuroHax

## Запуск тестов

```bash
# Все тесты
python -m pytest tests/ -v

# Конкретный модуль
python -m pytest tests/test_memory.py -v
python -m pytest tests/test_ppo.py -v

# Конкретный тест
python -m pytest tests/test_ppo.py::TestPPO::test_clip_ratio -v

# С покрытием
python -m pytest tests/ --cov=AI --cov-report=html
```

## Покрытие тестами

### ✅ Покрыто тестами:

#### `tests/test_memory.py` (10 тестов)
- `Memory.store()` — сохранение опыта
- `Memory.clear()` — очистка памяти (все поля)
- `Memory.copy_to_tensors()` — конвертация в тензоры
- Обработка truncated/terminal флагов
- Рост памяти при добавлении

#### `tests/test_ppo.py` (12 тестов)
- `PPO.compute_returns_and_advantages()` — GAE алгоритм
  - Terminal states
  - Truncated states
  - Все terminal
- `PPO.get_entropy_coef()` — затухание entropy
- `PPO.get_K_epochs()` — уменьшение эпох
- `PPO.update_combined()` — обновление политики
- `PPO.clip_ratio` — clip ratio в policy loss
- `PPO.value_loss_computation` — вычисление value loss
- Интеграционные тесты полного цикла

### ⚠️ Частично покрыто:

#### `tests/test_collector.py`
- Структура опыта
- Объединение данных агентов
- **Требует доработки**: тесты shared memory

### ❌ Не покрыто (требует моков):

#### `tests/test_environment.py`
- `Environment._compute_potentials()`
- `Environment.improved_rewards()`
- `Environment.step()`
- **Проблема**: требует `Player_actions.Net_action`

#### `tests/test_translator.py` (не создан)
- `Translator.translate_input()`
- Симметрия для team1/team2
- **Требуется**: тесты на симметрию

## Добавление новых тестов

1. Создайте файл `tests/test_<module>.py`
2. Импортируйте тестируемый модуль
3. Используйте pytest fixtures для setup
4. Добавляйте_assert для проверки

Пример:
```python
import pytest
from AI.YourModule import YourClass

class TestYourClass:
    @pytest.fixture
    def instance(self):
        return YourClass()
    
    def test_something(self, instance):
        result = instance.method()
        assert result == expected
```

## Статус

- **Memory**: ✅ 10/10 тестов
- **PPO**: ✅ 12/12 тестов  
- **Collector**: ⚠️ 0/6 (требует моков)
- **Environment**: ⚠️ 0/11 (требует моков)
- **Translator**: ❌ не создан
