# Team and Topic Selection

## Topic: Soul-like Gaming AI

## Team Members
- **Member 1:** YuTong Zhang
- **Member 2:** WanChi Kao
- **Member 3:** Zeyu Chen

## Abstract
### **Problem/Goal**
Our project aims to develop a reinforcement learning agent to defeat specific bosses in soul-like games. Soul-like games are focused on skill combinations and choices, precise execution, and sharp judgment. Training an agent to beat the final boss of Hollow Knight: Silksong, "$Lost Lace$", is our aim.

### **Method**
We plan to implement a $Deep$ $Q-Network(DQN)$ architecture as a starting point to train our agent through trial-and-error gameplay. If it turns out that a single $DQN$ is not capable of carrying out with this task, we will try $Proximal$ $Policy$ $Optimization(PPO)$. The agent will learn through repeated gameplay to know the game status, operations, boss health, and boss skills. Our training environment will include two mods, $ShowDamage - HealthBar$ mod to show the boss's health, and $Stakes of Marika - Rebirth Anywhere$ for instantly reviving the playable character to speed up training.

In order to provide the in-game information:
  - As the $ShowDamage - HealthBar$ mod provides precise health value(e.g. $1800/1800$) in black with white background, We plan to use $Optical$ $Character$ $Recognition(OCR)$ to read the boss's health 
  - As the boss fight background is in dark color while the player's health are shown in white, we plan to convert the player's health area(in the left top corner of the screen) to grayscale and denoise it.
  - As for the rewarding system, we plan to reward the agent by:
    -  Successfully hitting the boss.
    -  Every second survived in the boss fight.
    -  Win the boss fight(boss has 0 health).
 -  As for the punishing system, we plan to punish the agent by:
    -  Getting hit by the boss.
    -  Not hitting the boss in a certain amount of time(to avoid the agent from just running away from boss)
    -  Lose the boss fight(player has 0 health).

(We plan to implement a Deep Q-Network (DQN) as our starting approach, and later compare against or refine with Proximal Policy Optimization (PPO) to improve stability and performance.)

### **Challenge**
Recently, the long-awaited Hollow Knight: Silksong was finally released. This game was chosen as the foundation for our Souls-like AI training. Using a newly published game means we are dealing with brand new stuff, with no current AI solutions or proven techniques to draw on, making the project more demanding and worthwhile. Meanwhile, souls-like games are known for their high level of difficulty. From the game's release till September 10th, this game has sold over 5 million copies across all platform. However, only $0.8%$ of the players on Steam(one of the platforms) beat the final boss $Lost Lace$. This boss has $3$ different phases, over $10$ different skills, which will derive into numerous combos, which will make it harder for the agent to learn the tricks when fighting this boss.

### **Application**
This reinforcement learning agent can not only be used in fighting $Lost Lace$, but also be broaden to other types of games with few adjustments of code(including real world simulator games with appropriate rewarding and punishing system). Thus, this training structure can be further used to simulate real world scenarios with proper settings and hence resolving real world issues.