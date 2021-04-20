# Tennis-Match-Prediction
![Forks](https://img.shields.io/github/forks/shukkkur/Tennis-Match-Prediction.svg)
![Stars](https://img.shields.io/github/stars/shukkkur/Tennis-Match-Prediction.svg)
![Watchers](https://img.shields.io/github/watchers/shukkkur/Tennis-Match-Prediction.svg)
![Last Commit](https://img.shields.io/github/last-commit/shukkkur/Tennis-Match-Prediction.svg) 

Using different linear models I tried to predict the winner of a tennis matches based on 2 features.
<br><br>

Sample data: 

![sample](https://github.com/shukkkur/Tennis-Match-Prediction/blob/779f3cf1628611e25680701551f287af6ab728ec/.readme/sample.png)
<br><br>

<p>Since the only feautres available for predicting are the personal scores of the players (lk1 & lk2), <br >plotting them reveals a <b>linear relationship</b></p>

```python
sns.lineplot(x = 'lk1', y = 'lk2', data = dfs, hue = 'match_outcome')
plt.show()
```

![sample](https://github.com/shukkkur/Tennis-Match-Prediction/blob/ec366e3064af51c8e73ae0784d61d713a332f4a9/.readme/image.png)
<br><br>
I build and trained the following linear model classifiers:
- Ridge 
- Lasso
- KNN
- Logistic Regression
- Decision Tree

Then, tested them on the new ![datasets](PredictionData)

<br>
and the results I've got:

|    | date_of_match | player1_id | player1_name | player2_id |      player2_name     |  lk1 |  lk2 | final_outcome | Ridge - L2 |  SVC |  KNN | LogReg | DecisionTree |
|:--:|:-------------:|:----------:|:------------:|:----------:|:---------------------:|:----:|:----:|:-------------:|:----------:|:----:|:----:|:------:|:------------:|
|  0 | 27.09.2020    | 20754594   | Fischer      | 29753705   | Pfau                  | 15.1 | 15.1 | lose          | win        | lose | lose | win    | lose         |
|  1 | 27.09.2020    | 20754594   | Fischer      | 20355195   | Kösters               | 15.1 | 20.1 | win           | win        | win  | win  | win    | win          |
|  2 | 13.09.2020    | 20754594   | Fischer      | 28803686   | Opalka                | 15.1 | 22.1 | win           | win        | win  | win  | win    | win          |
|  3 | 05.09.2020    | 20754594   | Fischer      | 20457747   | Goedecke              | 15.1 | 15.1 | lose          | win        | lose | lose | win    | lose         |
|  4 | 23.08.2020    | 20754594   | Fischer      | 20357314   | van Raay              | 15.1 | 23.1 | win           | win        | win  | win  | win    | win          |
|  5 | 16.08.2020    | 20754594   | Fischer      | 20263300   | Verheyen              | 15.1 | 23.1 | win           | win        | win  | win  | win    | win          |
|  6 | 09.08.2020    | 20754594   | Fischer      | 29050113   | Siebert               | 15.1 | 22.1 | win           | win        | win  | win  | win    | win          |
|  7 | 12.07.2020    | 20754594   | Fischer      | 20651971   | Menze                 | 15.1 | 11.7 | lose          | lose       | lose | lose | lose   | win          |
|  8 | 09.07.2020    | 20754594   | Fischer      | 20551904   | Bodem                 | 15.1 | 15.1 | lose          | win        | lose | lose | win    | lose         |
|  9 | 09.07.2020    | 20754594   | Fischer      | 20653485   | Rajapreyar            | 15.1 | 8.0  | lose          | lose       | lose | lose | lose   | lose         |
| 10 | 29.06.2020    | 20754594   | Fischer      | 20556878   | Grigorieva            | 15.1 | 9.7  | lose          | lose       | lose | lose | lose   | lose         |
| 11 | 25.06.2020    | 20754594   | Fischer      | 20652120   | Götz                  | 15.1 | 19.1 | win           | win        | win  | win  | win    | win          |
| 12 | 25.06.2020    | 20754594   | Fischer      | 20651858   | Kriegbaum             | 15.1 | 21.8 | win           | win        | win  | win  | win    | win          |
| 13 | 25.06.2020    | 20754594   | Fischer      | 20653608   | Hein                  | 15.1 | 13.1 | lose          | lose       | lose | lose | win    | lose         |
| 14 | 21.06.2020    | 20754594   | Fischer      | 20652121   | Pohl                  | 15.1 | 16.8 | win           | win        | win  | win  | win    | win          |
| 15 | 19.06.2020    | 20754594   | Fischer      | 20651901   | Schaefer              | 15.1 | 6.1  | lose          | lose       | lose | lose | lose   | lose         |
| 16 | 12.06.2020    | 20754594   | Fischer      | 20652120   | Götz                  | 15.1 | 19.1 | win           | win        | win  | win  | win    | win          |
| 17 | 11.06.2020    | 20754594   | Fischer      | 20352348   | Kroll                 | 15.1 | 8.1  | lose          | lose       | lose | lose | lose   | lose         |
| 18 | 24.02.2020    | 20754594   | Fischer      | 20651370   | Wolter                | 15.1 | 12.1 | lose          | lose       | lose | win  | lose   | win          |
| 19 | 24.02.2020    | 20754594   | Fischer      | 20655872   | Bongardt              | 15.1 | 19.0 | win           | win        | win  | win  | win    | win          |
| 20 | 11.02.2020    | 20754594   | Fischer      | 20652388   | Buß                   | 15.1 | 4.1  | lose          | lose       | lose | lose | lose   | lose         |
| 21 | 11.02.2020    | 20754594   | Fischer      | 20650020   | Kuhlwein von Rathenow | 15.1 | 13.6 | lose          | lose       | lose | lose | win    | lose         |
