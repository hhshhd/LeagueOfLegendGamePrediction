# Prediction behind LOL
**Name**: Housheng Hai

A project designed to build and train a model for predicting LOL games result. DSC80 at UCSD.

### Framing the Problem
#### Problem Identification

The data set I am going to study is the game data of professional players in various League of Legends matches. The data set contains match data from the LCS, LEC, LCK, LPL, PCS, CBLoL, and many more leagues. (the structure of code below for data cleaning is similar to that of project 3, but with different columns and logic behind.)

I will choose ['gameid', 'position', 'result', 'side', 'firstblood', 'firstdragon', 'firsttower', 'golddiffat10', 'golddiffat15'] as my columns relevant to my question.

| Column_Name | Description |
| ----------- | ----------- |
| 'gameid | the id of each game/match. Each 'gameid' corresponds to up to 12 rows – one for each of the 5 players on both teams and 2 containing summary data for the two teams. |
| Paragraph | Text |
| 'position' | the role of the player in each match (top,jng, mid, bot, sup) |
| 'result' | WIN/LOSE (1 represents win in raw data) |
| 'side' | side of the team in the game, (Blue/Red) |
| 'firstblood' | whether the team/player contributed in the first kill of the game. (1 refers to True) |
| 'firstdragon' | whether the team kill the first dragon in the game. (1 refers to True) |
| 'firsttower' | whether the team destroy the first turret in the game. (1 refers to True)|
| 'golddiffat10' | Average gold difference at 10 minutes |
| 'golddiffat15' | Average gold difference at 15 minutes |

