import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clearml import Task
from clearml import Logger

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

task = Task.init(project_name="first_try", task_name="third_task")
logger = Logger.current_logger()

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")

train.drop(columns="product_code", inplace=True)
test.drop(columns="product_code", inplace=True)

target = np.ravel(train[['failure']])
df = train.drop(columns=["failure"])

cols = df.columns
cat_columns = [col for col in cols if df[col].dtypes == object]
num_columns = [col for col in cols if df[col].dtypes != object]

for col in num_columns:
    df[col] = df[col].fillna(df[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    df,
    target,
    test_size=0.2,
    random_state=42
)

feature_names = list(df.columns)

train_data = Pool(
    data=X_train,
    label=y_train,
    cat_features = cat_columns,
    feature_names = feature_names
)

eval_data = Pool(
    data=X_test,
    label=y_test,
    cat_features = cat_columns,
    feature_names=feature_names
)

scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])

model = CatBoostClassifier(
    iterations = 1000,
    early_stopping_rounds=100,
    verbose = 100,
    cat_features = cat_columns,
    eval_metric= 'AUC',
    scale_pos_weight=6
)

model.fit(X=train_data, eval_set=eval_data)

y_predict=model.predict(eval_data)

logger.report_table(
    title="classification_report",
    series='pandas DataFrame',
    table_plot=pd.DataFrame(classification_report(y_test, y_predict, target_names=["0", "1"], output_dict=True))
)

predict=model.predict(test)

submission['failure'] = predict

submission.to_csv("data/sample_submission.csv", index=False)

plt.figure(figsize=(4,4))
logger.report_confusion_matrix(
    title="Confusion matrix",
    series="ignored",
    matrix=confusion_matrix(y_test, y_predict),
)