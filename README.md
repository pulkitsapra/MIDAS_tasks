# MIDAS_tasks

- This Repo has 2 folders, one containing python problem and other containing CV Problem

### Python Problem Assumptions

- Retweets and Reply Tweets are considered to be valid tweets. 

### CV Problem

- Initially data was visualized by plotting the train images.
- Various ML classifiers were tried from Scikit Learn namely Naive Bayes, SVM, KNN, MLP Classifier,  QuadraticDiscriminantAnalysis and Random Forest
- Maximum performance on validation set was around 81.75 % without Data Augmentation and 82.85 % with data augmentation for Randomforest (n_estimators=100, max_depth=10)
- CNN Model-1 beats this performance as accuracy is close to 88 % on validation set.
