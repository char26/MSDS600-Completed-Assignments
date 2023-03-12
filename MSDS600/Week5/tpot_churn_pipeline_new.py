import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv(r'C:\Users\Charlie\Desktop\Charlie\GitHub\MSDS600\Week5\Assignment\prepped_churn_data.csv', index_col=0)
features = tpot_data.drop('Churn', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn'], random_state=26)

# Average CV score on the training set was: 0.7969290538413807
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.25, min_samples_leaf=3, min_samples_split=7, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 26)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(results)
