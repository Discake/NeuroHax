"""
Конфигурация pytest для тестов NeuroHax
"""
import sys
import os
import pytest
from unittest.mock import MagicMock

# Добавляем корень проекта в sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Создаём моки для отсутствующих модулей перед импортом Collector
missing_modules = [
    'Player_actions',
    'Player_actions.Net_action',
    'AI.Training.Environment',
]

for module_name in missing_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

# Фикстура для временной директории
@pytest.fixture
def temp_dir():
    """Создаёт временную директорию и очищает после теста"""
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    os.chdir(test_dir)
    
    yield test_dir
    
    os.chdir(original_cwd)
    shutil.rmtree(test_dir, ignore_errors=True)
