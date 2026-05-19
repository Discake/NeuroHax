@echo off
:: Fine-tuning SimpleModel на основе существующего чекпоинта
:: Укажи путь к чекпоинту в --weights
:: --start-episode должен соответствовать номеру, на котором остановилось обучение

set WEIGHTS=models/simple_model_vs_37.pth
set START_EP=0
set EPISODES=100
set SAVE=models/simple_model_vs_39.pth
set POOL=models/simple_model_vs_1.pth

python train_simple_model_workers.py ^
    --weights %WEIGHTS% ^
    --start-episode %START_EP% ^
    --episodes %EPISODES% ^
    --workers 8 ^
    --steps 4096 ^
    --save %SAVE% ^
    --save-interval 10 ^
    --phase1-frac 0.1 ^
    --opponent-sync 100
    @REM --pool-models %POOL%

pause
