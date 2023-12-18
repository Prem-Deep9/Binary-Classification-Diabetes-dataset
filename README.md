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

# Guide for running the code
1. Install Required Libraries: Navigate to the project directory and install the required libraries using the provided requirements.txt file. Run the following command in your terminal or command prompt:
`pip install -r requirements.txt`
2. Update Dataset Path in the Notebook: Look for the cell Reading Data, that reads the dataset (e.g., diabetes.xlsx). There should be a line of code that looks like this:\
Diabetes_Data= pd.read_excel('diabetes.xlsx')\
Change the path within the quotation marks to the correct path on your PC. For example:\
Diabetes_Data = pd.read_excel("C:/Users/YourUsername/Documents/dataset/diabetes.xlsx")
3. Run the Cells: Run each cell in the Jupyter Notebook sequentially by pressing Shift + Enter. Make sure to execute any cells that load or preprocess the data.
4. Verify Results: Check the output of the cells to ensure that the code runs without errors, and the dataset is loaded successfully.\
### Note:
Make sure to have Python and Jupyter Notebook installed on your machine.

# Results
The evaluation results demonstrate that the Mean replacement of NaN values tends to be a better model than KNN Imputer, Furthermore, Random Forest performs better than KNN and SVM. The following table summarizes the performance metrics for each classification algorithm:

| Algorithm    | Accuracy | Precision | Recall | F1-Score |
| ------------ | -------- | --------- | ------ | -------- |
| Random Forest | 93.75    | 91.66     | 93.90  | 92.77    |
| KNN          | 90.62    | 84.78     | 95.12  | 89.65    |
| SVM          | 85.41    | 84.61     | 80.48  | 82.50    |

These results suggest that, for the given dataset and task, Mean replacement for imputation and Random Forest for classification yield the most favorable outcomes.

# Inference of results:
- Based on the results the ideal method to choose for NaN Replacement is by using Mean, and the best algorithm to use is Random Forest based on accuracy or F-1Score (because data is balanced accuracy should be given higher consideration).
- SVM, KNN, and Random Forest are selected due to their capabilities as powerful nonlinear classifiers.
- Even though SVM uses a polynomial kernel it does not perform well(relatively) with the dataset as data has noise, although it is suitable for higher-dimensional datasets, it underperforms as there is no clear margin of separation.
- KNN is one of the best algorithms for nonlinear classification and performs well on the data set it works well for noisy data as only the neighboring data matters for its classification, the only drawback is finding the optimal value of k which is accomplished using Grid Search.
- Random Forest is one of the best classifiers for continuous numeric data (which our data is), also it can handle higher dimensions that fit the model the best, having a close call with KNN.

# Challenges of the project:
- Handling missing values has been the first challenge, have come across multiple solutions to handle null values, had to try out each to see if it is overfitting or underfitting, and also thinking logically if the approach of cleaning data in a particular way is correct.
- The option of using various other techniques like cross-validation, feature extraction, and variations of balancing the dataset to try and improve the accuracies. Even though we have achieved good accuracy values, there was always a sense of doubt about the inclusion of a technique that could make it better.
- Balancing datasets, outlier removal, and parameter tuning have been minor challenges until figured out how to implement them.

# Future work
Various approaches to clean the data have been identified, including deletion of null rows, selective deletion, utilization of mean, median, or mode, feature extraction, duplication of similar data, and the use of imputer algorithms. These methods can also be employed in combination for comprehensive data cleaning. Additionally, it has been recognized that several supplementary steps, such as feature extraction through multiple methods, balancing classes (via over-sampling, under-sampling, SMOTE, etc.), early stopping, and cross-validation, can be incorporated to enhance the accuracies of our dataset. To gain a foundational understanding of the effectiveness of these methods, we could apply them to different datasets and analyze their performance, identifying which approaches are particularly well-suited for specific types of datasets.

