import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import time
import sys
import os
import memory_profiler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.append(os.path.abspath("/projects/genomic-ml/da2343/ml_project_1/shared"))
from model_header import *
from constants import *

import warnings
warnings.filterwarnings("ignore")


# Generate synthetic regression data
np.random.seed(42)
total_samples = 1_000_000
X = np.random.rand(total_samples, 4)  # Assuming 4 features for each sample
y = 10 * X[:, 0] + 5 * X[:, 1] - 2 * X[:, 2] + np.random.normal(0, 1, total_samples)  # Linear regression example
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# iris = datasets.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

threshold_param_list = np.arange(0, 1.01, 0.1)
threshold_param_dict = [
    {"threshold": [threshold]} for threshold in threshold_param_list
]

alpha_param_list = [10**x for x in range(-10, 1)]
alpha_param_dict = [{"alpha": [alpha]} for alpha in alpha_param_list]

@memory_profiler.profile
def featureless(N):
    clf = Featureless()
    start_time = time.time()
    N_X_train = X_train[:N]
    N_X_test = X_test[:N]
    N_y_train = y_train[:N]
    clf.fit(N_X_train, N_y_train)
    end_time = time.time()
    predictions = clf.predict(N_X_test)
    return predictions


@memory_profiler.profile
def spearman_rank_regressor(N):
    clf = GridSearchCV(
        SpearmanRankRegressorTest(),
        threshold_param_dict,
        scoring="neg_mean_squared_error",
        return_train_score=True
    )
    start_time = time.time()
    N_X_train = X_train[:N]
    N_X_test = X_test[:N]
    N_y_train = y_train[:N]
    clf.fit(N_X_train, N_y_train)
    end_time = time.time()
    predictions = clf.predict(N_X_test)
    return predictions


@memory_profiler.profile
def my_pearson_regressor(N):
    clf = GridSearchCV(
        MyPearsonRegressor(),
        threshold_param_dict,
        scoring="neg_mean_squared_error",
        return_train_score=True
    )
    start_time = time.time()
    N_X_train = X_train[:N]
    N_X_test = X_test[:N]
    N_y_train = y_train[:N]
    clf.fit(N_X_train, N_y_train)
    end_time = time.time()
    predictions = clf.predict(N_X_test)
    return predictions

@memory_profiler.profile
def lasso_cv(N):
    clf =  LassoCV(random_state=1)
    start_time = time.time()
    N_X_train = X_train[:N]
    N_X_test = X_test[:N]
    N_y_train = y_train[:N]
    clf.fit(N_X_train, N_y_train)
    end_time = time.time()
    predictions = clf.predict(N_X_test)
    return predictions

@memory_profiler.profile
def ggm(N):
    clf = GaussianGraphicalModel()
    start_time = time.time()
    N_X_train = X_train[:N]
    N_X_test = X_test[:N]
    N_y_train = y_train[:N]
    clf.fit(N_X_train, N_y_train)
    end_time = time.time()
    predictions = clf.predict(N_X_test)
    return predictions


def measure_time_and_memory(N_values, functions, num_iterations=5):
    data = {'N': [], 'Function': [], 'Mean Time': [], 'Std Time': [], 'Mean Memory': [], 'Std Memory': []}

    for func in functions:
        for N in N_values:
            time_measurements = []
            memory_measurements = []

            for _ in range(num_iterations):
                # Measure time
                start_time = time.time()
                func(N)
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_measurements.append(elapsed_time)

                # Measure memory
                memory_measurements.append(
                    np.max(memory_profiler.memory_usage(
                        (func, (N,), {})
                    ))
                )

            # Calculate mean and standard deviation
            mean_time = np.mean(time_measurements)
            std_time = np.std(time_measurements)

            mean_memory = np.mean(memory_measurements)
            std_memory = np.std(memory_measurements)

            data['N'].append(N)
            data['Function'].append(func.__name__)
            data['Mean Time'].append(mean_time)
            data['Std Time'].append(std_time)
            data['Mean Memory'].append(mean_memory)
            data['Std Memory'].append(std_memory)

    return pd.DataFrame(data)

if __name__ == "__main__":
    # Set the range of input sizes (N)
    N_values = [10, 10**2, 10**3, 10**4, 10**5, 10**6]
    # Set the number of iterations for each input size
    num_iterations = 3
    # Define example functions to test
    functions_to_test = [featureless, spearman_rank_regressor, my_pearson_regressor, lasso_cv, ggm]
    # Measure time and memory
    results_df = measure_time_and_memory(N_values, functions_to_test, num_iterations)
    # Display the DataFrame
    print(results_df)
    
    # save the results
    results_df.to_csv("results.csv", index=False)