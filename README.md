# Soul-like-Gaming-AI
This repo will utilize reinforcement learning techniques to train an AI to play soul-like games, such as Hollow Knight(potentially including Silksong), to beat a specific boss.
Using PPO to play Hollow Knight Silksong to beat the boss Lace in Cradle.

## Game Settings
- Resolution: 1280 x 720 @60.02HZ
- Window mode (disable fullscreen)
- Window position: Top-left corner

## Mods
The mods used for training here are:
1. https://thunderstore.io/c/hollow-knight-silksong/p/BepInEx/BepInExPack_Silksong/ Silksong mod manager. Dependency for the following 2 mods.
2. https://thunderstore.io/c/hollow-knight-silksong/p/XiaohaiMod/ShowDamage_HealthBar/ For OCR to read the boss health.
3. https://www.nexusmods.com/hollowknightsilksong/mods/46 Revive instantly to speed up training.

Modify the ShowDamage-HealthBar mod, changed the boss HP threshold from 1000 to 500 to show the HP bar.

## Boss
- Name: Lace
- HP: 800
- Location: The Cradle

## Files
- `calibrate_boss_hp.py`
- `calibrate_player_hp.py`
- `boss_hp_detector.py`
- `player_hp_detector.py`
- PNGs about HP detection

## Train
```bash
python train.py
```
**Training steps**
1. Pre-exploration: 50 Episodes
2. Formal training: 100 Updates
3. Automatic saving of model and logs
4. Press **Ctrl+C** to stop if need

**Current training Parameters**
```
Observation size: 128×128
Frame Stack: 4
Number of actions: 10
Learning rate: 5e-4
Entropy: 0.1
Hit reward: 2.0
Injury penalty: 0.5
```
