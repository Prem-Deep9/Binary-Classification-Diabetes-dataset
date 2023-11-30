# Binary-Classification-Diabetes-dataset
Performance analysis of multiple supervised learning methods for solving a binary classification problem

# Introduction
Supervised learning is a subcategory of machine learning that uses labeled datasets to train algorithms to classify data or predict outcomes accurately. In this project, we are comparing the performance of the following three supervised learning algorithms (optimal Parameters were found using Grid Search, optimal parameters change every run)
### KNN
**Optimal Parameters:** `{metric: 'manhattan', n_neighbors: 19, weights: 'distance'}`

### Random Forest
**Optimal Parameters:** `{criterion: 'entropy', max_depth: 11, max_features: 'log2', n_estimators: 200}`

### SVM
**Optimal Parameters:** `{C: 3, kernel: 'linear', max_iter: 2000}`
### The following libraries have been used for the implementation of the project:
- Scikit-learn – To implement the various ML algorithms and other calculations
- Pandas – To read data and work with the data frames
- NumPy – Arrays and other data manipulations
- Matplotlib – For plotting graphs
- Seaborn – For plotting graphs
- Imblearn – for balancing the dataset

# Evaluation Describing Results:
Evaluation results show that the Mean replacement of NaN values tends to be a better model than KNN Imputer, Furthermore, Random Forest performs better than KNN and SVM.

# Conclusion of results:
Based on the results the ideal method to choose for NaN Replacement is by using Mean, and the best algorithm to use is Random Forest based on accuracy or F-1Score (because data is balanced accuracy should be given higher consideration).
SVM, KNN, and Random Forest are chosen because they are powerful nonlinear classifiers.
Even though SVM uses a polynomial kernel it does not perform well(relatively) with the dataset as data has noise, although it is suitable for higher-dimensional datasets it underperforms as there is no clear margin of separation.
KNN is one of the best algorithms for nonlinear classification performs well on the data set it works well for noisy data as only the neighboring data matters for its classification, the only drawback is finding the optimal value of k which is accomplished using Grid Search.
Random Forest is one of the best classifiers for continuous numeric data (which our data is), also it can handle higher dimensions that fit the model is the best, having a close call with KNN.

# Challenges of the project:
- Handling missing values has been the first challenge, have come across multiple solutions to handle null values, had to try out each to see if it is overfitting or underfitting, and also thinking logically if the approach of cleaning data in a particular way is correct.
- The option of using various other techniques like cross-validation, feature extraction, and variations of balancing the dataset to try and improve the accuracies. Even though we have achieved good accuracy values, there was always a sense of doubt about the inclusion of a technique that could make it better.
- Balancing datasets, outlier removal, and parameter tuning have been minor challenges until figured out how to implement them.

# Future work
Various approaches to clean the data (Deletion of null rows, selective deletion, use of mean, median, or mode, feature extraction, Duplication of similar data, and use of imputer algorithms) have been identified, and a few of them can be used in combination too. Similarly, have learned that multiple additional steps like feature extraction (multiple ways), balancing classes (over-sampling, under-sampling, SMOTE, etc), and cross-validation can be incorporated to improve the accuracies of our data set. We could run different datasets on the approaches discussed above to have a basic understanding of which methods are ideal for a particular kind of dataset.
