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

Then the dataframe with selected columns would be shown below,

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

|   position |
|-----------:|
|      10633 |
|      10629 |

Accuracy is straightforward to understand and interpret, and it will work well for my prediction model since there is an approximately equal number of instances for each class(Win/Lose) as when there is a team winning a game, there will always be another loser in this game. So there will always be equal number of instances for each class if the data is collected completely. Even with potential missingless of data collection, the number of these two classes will still be close. So the classes in the dataset are balanced, then accuracy can be a suitable metric to evaluate my model's performance.

Meanwhile, the columns, which mostly the features I plan to use for the model to predict, are all of them can be collected **before the mid-game teamfight**, which means these informations can be known at the "time of prediction". And thats why I clarify the specific time period in my prediction question, since without that, people may consider to use other data, such as final teamkills, gold difference at the end of the game, to predict the result. However, that might make our preidction question and model a bit of useless since we wanna predict things that havent happened with light clues, using final teamkills and other data that are recorded after the game has been finished to predict the result will make such progress meaningless. So like among the features I choose, 'side', 'firstblood', 'firstdragon', 'firsttower','golddiffat10/15' they all can be recorded/collected/noticed during the game/before the final result released, which do leave us space to guess/predict.

### Baseline Model

In this baseline model, I use a logistic regression classifier to predict the outcome of a game (win or lose) based on several features. 

Model Description:

    Classifier: Logistic Regression
    Features:
        Quantitative features: 'golddiffat10' (1 quantitative features)
        Categorical features:
            'side' (nominal, 2 categories: Blue, Red)
            'firstblood' (nominal, 2 categories: 0, 1)
            'firstdragon' (nominal, 2 categories: 0, 1)
            'firsttower' (nominal, 2 categories: 0, 1)
            
Encoding of Categorical Features:

One-Hot Encoding was used to encode the categorical features ('side', 'firstblood', 'firstdragon', 'firsttower'). This encoding technique creates binary columns for each category, indicating the presence or absence of that category.


Also, to ensure the model’s ability to generalize to unseen data, I apply train_test_split to split 20% of data randomly as the test set and remained as training set. Thus we can test the accuracy/performance of the model on unseen data by using the test set of data.

Performance of the Model:

I use Accuracy to evaluate the performance of the model (the reason of choosing Accuracy has been explained in previous section), which measures the proportion of correctly classified instances out of the total predictions. The Accuracy I got is 0.7166705854690807, approxmately 71.67%. With such accuracy, I believe the performance of this model is just ok as a baseline model. Since this model is able to classify the outcomes of the games with a certain level of accuracy, but 71% is not much perfect since guessing the outcome randomly without any model/infos provided will have approx 50% accuracy. This model only raised 50% of accuracy, which is not enough. My desired perfomance should at least reach 75%, which is close but still there's still a distance. This baseline model is a starting point and may not capture all complexities of the problem.
    
### Final Model

Based on baseline model, I add two new feature engineering steps in the pipeline: standard scaling of the 'golddiffat10' and 'golddiffat15' columns using StandardScaler, and applying quantile transformation to the 'golddiffat10' column using QuantileTransformer.

Then I use GridSearchCV to search for the best hyperparameters of the RandomForestClassifier. The hyperparameters tuned in this example are the number of estimators (n_estimators), maximum depth of trees (max_depth), and minimum number of samples required to split an internal node (min_samples_split).

The final model is trained on the whole dataset using the best hyperparameters obtained from the grid search. The accuracy of the final model is evaluated based on predictions made on the same dataset.

In the final model, I add **two** new features: 'golddiffat10' (standard scaled) and 'golddiffat15' (standard scaled). These features were chosen based on their potential relevance to the the outcome variable.

'golddiffat10': This feature represents the difference in gold between the two teams at the 10-minute mark. It captures the early game advantage or disadvantage of a team. By standard scaling this feature, we bring it to a common scale, which can help in achieving better model performance.

'golddiffat15': Similar to 'golddiffat10', this feature represents the difference in gold between the two teams, but at the 15-minute mark. It provides additional information about the relative strength or weakness of the teams, and especially shows the trend whether the team is expanding their strength in last 5 minutes or waste their leading outcomes. I also apply standard scaling on this feature hope that may achieve better model performance as golddiffat10 does.

These features are strong relevant because the gold difference at specific time intervals(espcially the beginning of the game) can be indicative of a team's performance and the likelihood of winning a game. The logic behind is that  having more golds at the beginning will have better equipment earlier, so it will be easier to gain an advantage in numerical values when fighting with the enemy, thereby expanding the acquisition of more resources, so as to win.  The information captured by these features reflects the dynamics and progress of the game, making them potentially informative for predicting the outcome.

In the above baseline model, I use a **RandomForestClassifier** to predict the outcome of a game (win or lose) based on several features. Random Forest is an ensemble algorithm that combines multiple decision trees to make predictions. It is known for its ability to handle complex relationships and capture non-linear patterns in the data. Because most of the features I selected do not have a strong direct relationship with the outcome, so I want to obtain non-linear relationships through the decision tree.

Hyperparameters are tuned using **GridSearchCV** with 5-fold cross-validation. The hyperparameters tuned in this example are:

    n_estimators: The number of trees in the random forest.

    max_depth: The maximum depth of each decision tree.

    min_samples_split: The minimum number of samples required to split an internal node.

The best hyperparameters obtained from the grid search are:

    n_estimators: 200

    max_depth: 5

    min_samples_split: 2


To select the hyperparameters for the final model, we used the method of grid search combined with cross-validation. First, I define a grid of hyperparameters('n_estimators': [50, 100, 200],'max_depth': [None, 5, 10],'min_samples_split': [2, 5, 10]) for RandomForestClassifier. Then I use the GridSearchCV class, which performs searches over the specified hyperparameter grid while utilizing cross-validation, fitting the grid search object to the training data, which triggers the exploration of different combinations of hyperparameters and evaluation of model performance using cross-validation. After the grid search is completed, the best model and the best hyperparameters are available through best_model and best_params_.

The final model's accuracy of approximately 73.90% is a bit higher than the accuracy of the baseline model (71.67%), which shows an improvement in performance compared to the baseline model. This improvement in accuracy suggests that the final model is better at correctly classifying the outcomes of the games. It indicates that the **additional features** engineered and the **optimized hyperparameters** helped enhance the model's predictive ability.

With incorporating more relevant features, the model can capture more nuanced patterns in the data.

With optimized hyperparameters, it enhances the flexibility of the model to capture complex relationship.

<iframe src="assets/Figure-1.png" width=800 height=600 frameBorder=0></iframe>

The above figure is a confusion matrix provides a visual representation of the model's performance, showing the number of true positives, true negatives, false positives, and false negatives. We can clearly see that the dark proportion of the entire data is approx 75%, which do shows an improvment in permofance of the model.

### Fairness Analysis

To perform a fairness analysis of the final model, I assume two groups are:

    Group A: Blue team (side = 'Blue')

    Group B: Red team (side = 'Red')

For the evaluation metric, I choose to use **Accuracy**, which measures the overall correct predictions of the model. As I explained previously, since my dataset and classes are balanced, where numbers of win and lose are approxmately the same, so Accuracy will works just fine.

Then, let's perform the permutation test and state the null and alternative hypotheses:

    Null Hypothesis: Our model is fair. It's accuracy for the Blue team and the Red team is roughly the same, and any differences are due to random chance.

    Alternative Hypothesis: Our model is unfair. The accuracy for the Blue team is lower than the accuracy for the Red team.
    
For the choice of test statistic, We need one that can measure how different the model's performance between two groups (X and Y), so I choose **the difference in accuracy between the two groups** as the test statistic

For significane level, I'll choose one of common used one, **5%**, as the the probability, under the null hypothesis, that the test statistic is equal to the value that was observed in the data or is even further in the direction of the alternative.

After running our permutation test, I get

`Observed Difference: -0.0019753550935941444`
`p-value: 0.792`
`Conclusion: Fail to reject the null hypothesis.`

<iframe src="assets/figure-1.html" width=800 height=600 frameBorder=0></iframe>

By looking at the graph and two lines(red represents observed stats and purple represents significance level of 5%, or by looking at the p_value = 0.792, thus we fail to reject that the null hypothesis that the accuracy for the Blue team and the Red team is roughly the same.

This indicates that there is no significant difference in accuracy between the Blue team (Group X) and the Red team (Group Y). Based on this analysis, we do not have sufficient evidence to conclude that the model is unfair in terms of its accuracy performance for the two groups.

