import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
from scipy.stats import pearsonr
from log_code import setup_logging
logger = setup_logging("fs")
from sklearn.feature_selection import VarianceThreshold


def select_best_columns(X_train, X_test, y_train, y_test):
    try:
        logger.info(f"Before constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"Before constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        # constant Technique
        var_obj = VarianceThreshold(threshold=0.0)
        var_obj.fit(X_train)

        logger.info(f"Columns to remove : {X_train.columns[~var_obj.get_support()]}")

        constant_columns = X_train.columns[~var_obj.get_support()]

        X_train = X_train.drop(constant_columns, axis=1)
        X_test = X_test.drop(constant_columns, axis=1)

        logger.info(f"After constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        # quasi Constant
        quasi_obj = VarianceThreshold(threshold=0.1)
        quasi_obj.fit(X_train)

        logger.info(f"Columns to remove : {X_train.columns[~quasi_obj.get_support()]}")

        quasi_constant_columns = X_train.columns[~quasi_obj.get_support()]

        X_train = X_train.drop(quasi_constant_columns, axis=1)
        X_test = X_test.drop(quasi_constant_columns, axis=1)

        logger.info(f"After Quasi constant Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After Quasi constant Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        # correlation with Hypothesis Testing
        # corr_p_values = []
        # for i in X_train.columns:
        #     values = pearsonr(X_train[i], y_train)
        #     corr_p_values.append(values)
        #
        # corr_p_values = np.array(corr_p_values)
        # p_values = corr_p_values[:, 1]
        # plt.figure(figsize=(5, 3))
        # plt.title("Hypothesis Testing")
        #
        # plt.xlabel("column Names")
        # plt.ylabel("P_values for Each Independent column")
        #
        # plt.bar(X_train.columns, p_values)
        #
        # plt.show()

        logger.info(f"After Hypothesis Technique X_train columns and shape : {X_train.columns} : {X_train.shape}")
        logger.info(f"After Hypothesis Technique X_test columns and shape : {X_test.columns} : {X_test.shape}")

        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} : due to : {er_type} : reason : {er_msg}")