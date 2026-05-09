@echo off
:: Обучение SimpleModel с нуля
:: Фаза 1 (0-20%% эпизодов): замороженный случайный оппонент
:: Фаза 2 (20-100%%): self-play с лаговым оппонентом

python train_simple_model_workers.py ^
    --episodes 500 ^
    --workers 8 ^
    --steps 1024 ^
    --save models/simple_model_vs.pth ^
    --save-interval 10 ^
    --lr 1e-3 ^
    --phase1-frac 0.20 ^
    --opponent-sync 10

pause
