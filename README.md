# Soul-like-Gaming-AI
This repo will utilize reinforcement learning technique to train an AI to play soul-like games such as Hollow Knight(potentially including silksong)  to beat a specific boss.

We haven't settle down which game specifically to let the AI play and what architecture to use. Currently thinking using DQN to play Hollow Knight Silksong to beat the final boss Lace in Abyss.
THe mods used for training here are:
1. https://thunderstore.io/c/hollow-knight-silksong/p/BepInEx/BepInExPack_Silksong/ Silksong mod manager. Dependency for the following 2 mods.
2. https://thunderstore.io/c/hollow-knight-silksong/p/XiaohaiMod/ShowDamage_HealthBar/ For OCR to read the boss health.
3. https://www.nexusmods.com/hollowknightsilksong/mods/46 Revive instantly to speed up training.
Fingers-crossed to pray it will work XD.

XD Obviously it's not working XD.
1. Cannot correctly determine terminate state
2. Cannot correctly read boss health
3. thus no efficient learning conducted >_<
