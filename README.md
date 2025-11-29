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
