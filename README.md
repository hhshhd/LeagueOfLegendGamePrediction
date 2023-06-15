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
| 'position' | the role of the player in each match (top,jng, mid, bot, sup) |
| 'result' | WIN/LOSE (1 represents win in raw data) |
| 'side' | side of the team in the game, (Blue/Red) |
| 'firstblood' | whether the team/player contributed in the first kill of the game. (1 refers to True) |
| 'firstdragon' | whether the team kill the first dragon in the game. (1 refers to True) |
| 'firsttower' | whether the team destroy the first turret in the game. (1 refers to True)|
| 'golddiffat10' | Average gold difference at 10 minutes |
| 'golddiffat15' | Average gold difference at 15 minutes |

| gameid                | position   |   result | side   |   firstblood |   firstdragon |   firsttower |   golddiffat10 |   golddiffat15 |
|:----------------------|:-----------|---------:|:-------|-------------:|--------------:|-------------:|---------------:|---------------:|
| ESPORTSTMNT01_2690210 | top        |        0 | Blue   |            0 |           nan |          nan |             52 |            391 |
| ESPORTSTMNT01_2690210 | jng        |        0 | Blue   |            1 |           nan |          nan |            485 |            541 |
| ESPORTSTMNT01_2690210 | mid        |        0 | Blue   |            0 |           nan |          nan |            162 |           -475 |
| ESPORTSTMNT01_2690210 | bot        |        0 | Blue   |            1 |           nan |          nan |            296 |           -793 |
| ESPORTSTMNT01_2690210 | sup        |        0 | Blue   |            1 |           nan |          nan |            528 |            443 |

Since in this project we only consider the prediction of team's result, so I'll filter my dataset to only keep the rows of teams status, "position" = team.

| gameid                | position   |   result | side   |   firstblood |   firstdragon |   firsttower |   golddiffat10 |   golddiffat15 |
|:----------------------|:-----------|---------:|:-------|-------------:|--------------:|-------------:|---------------:|---------------:|
| ESPORTSTMNT01_2690210 | team       |        0 | Blue   |            1 |             0 |            1 |           1523 |            107 |
| ESPORTSTMNT01_2690210 | team       |        1 | Red    |            0 |             1 |            0 |          -1523 |           -107 |
| ESPORTSTMNT01_2690219 | team       |        0 | Blue   |            0 |             0 |            0 |          -1619 |          -1763 |
| ESPORTSTMNT01_2690219 | team       |        1 | Red    |            1 |             1 |            1 |           1619 |           1763 |
| 8401-8401_game_1      | team       |        1 | Blue   |            0 |           nan |          nan |            nan |            nan |

For the missingless of some data, I choose to simply drop all rows with missing values, since if we provide any imputation to replace these missing values, which could defintely become the feature that used to trained the sample, it will create a bias inside the model with the data we improvised and leading to a larger impact on our model with such bias. Also, when dropping all missing values, there is still more than 20k rows that can be used for our project analysis, so there is no need to apply imputation which may cause risks to our project.

| gameid                | position   |   result | side   |   firstblood |   firstdragon |   firsttower |   golddiffat10 |   golddiffat15 |
|:----------------------|:-----------|---------:|:-------|-------------:|--------------:|-------------:|---------------:|---------------:|
| ESPORTSTMNT01_2690210 | team       |        0 | Blue   |            1 |             0 |            1 |           1523 |            107 |
| ESPORTSTMNT01_2690210 | team       |        1 | Red    |            0 |             1 |            0 |          -1523 |           -107 |
| ESPORTSTMNT01_2690219 | team       |        0 | Blue   |            0 |             0 |            0 |          -1619 |          -1763 |
| ESPORTSTMNT01_2690219 | team       |        1 | Red    |            1 |             1 |            1 |           1619 |           1763 |
| ESPORTSTMNT01_2690227 | team       |        1 | Blue   |            0 |             1 |            1 |           -103 |           1191 |

My prediction problem is: **Predict if a team will win or lose a game before the mid-game teamfight.**

This is an example of a **classification** problem. The goal is to classify the outcome of the game into two distinct classes: "win" or "lose", and since there is only two outcomes, it is a **binary classification problem**. So in this case the response variable, the variable I will predict, is **'result'**, which clearly represents the final outcome of the team. The reason choosing 'result' is that it has the exact meaning we want for our prediction, the (Win/Lose) result of the game.

For the metric, I choose **Accuracy**, which measures the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. Calculation given below:

Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

By running code below we can clearly see that The number of win and that of lose are approxmately same.

`lol2022_team_filtered.groupby('result')['position'].count()`
result
0    10633
1    10629
Name: position, dtype: int64












