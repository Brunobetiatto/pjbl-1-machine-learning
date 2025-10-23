from algoritmos_class import train_and_test_classifier
from algoritmos_class import train_and_test_regressor


# === chamama dos modelos de regressão ===

res_knn_regression = train_and_test_regressor(
    model_name="knn",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_rf_regression = train_and_test_regressor(
    model_name="random_forest",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_linear_regression = train_and_test_regressor(
    model_name="linear_regression",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_decision_tree = train_and_test_regressor(
    model_name="decision_tree",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_mlp = train_and_test_regressor(
    model_name="mlp",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_svr = train_and_test_regressor(
    model_name="svm",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)   

res_ensemble_regression = train_and_test_regressor(
    model_name="ensemble",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_bagging_regression = train_and_test_regressor(
    model_name="bagging",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)



res_stacking_regression = train_and_test_regressor(
    model_name="stacking",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)



res_bagging_regression = train_and_test_regressor(     
    model_name="bagging",
    csv_path="datasets/IRIS.csv",
    feature_columns=['sepal_length','sepal_width','petal_length'],
    target_column='petal_width',
    random_state=42,
    model_params='auto',
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)


# == chamado de modelos de classificação ==

# === chama os dois modelos e guarda resultados ===
res_knn = train_and_test_classifier(
    model_name="knn",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
       'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
       'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
       'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'n_neighbors': 5},
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold',
)

res_log = train_and_test_classifier(
    model_name="logistic_regression",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
       'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
       'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
       'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'C': 1.0},
    train_size=0.65,
    verbose=True,
    compute_importance=False,
    validation_method='k_fold'
)

res_rf = train_and_test_classifier(
    model_name="random_forest",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
       'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
       'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
       'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'n_estimators': 100},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)

res_svm = train_and_test_classifier(
    model_name="svm",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'C': 1.0},
    train_size=0.65,
    verbose=True,
    compute_importance=False

)

res_arvore = train_and_test_classifier(
    model_name="decision_tree",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'max_depth': 5},
    train_size=0.65,
    verbose=True,
    compute_importance=False
) 
res_naive = train_and_test_classifier(
    model_name="naive_bayes",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',    
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'var_smoothing': 1e-9},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)

res_mlp = train_and_test_classifier(
    model_name="mlp",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'hidden_layer_sizes': (100,), 'activation': 'relu'},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)


res_ensemble = train_and_test_classifier(
    model_name="ensemble",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level', 
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'var_smoothing': 1e-9},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)


res_bagging = train_and_test_classifier(
    model_name="bagging",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'n_estimators': 100},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)

res_boosting = train_and_test_classifier(
    model_name="boosting",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    random_state=42,
    model_params={'n_estimators': 100},
    train_size=0.65,
    verbose=True,
    compute_importance=False
)

res = train_and_test_classifier(
    model_name="stacking",
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column="Sleep_Quality",
    model_params={
        'cv': 5,
        'stack_method': 'predict_proba',
        'passthrough': False,
        'n_jobs': -1,
    },
    compute_importance=False,
    verbose=True,
    validation_method='kfold',
)

res = train_and_test_classifier(
    model_name='blending',
    csv_path="datasets/synthetic_coffee_health_10000.csv",
    feature_columns=['Age', 'Gender', 'Country', 'Coffee_Intake', 'Caffeine_mg',
        'Sleep_Hours', 'BMI', 'Heart_Rate', 'Stress_Level',
        'Physical_Activity_Hours', 'Health_Issues', 'Occupation', 'Smoking',
        'Alcohol_Consumption'],
    target_column='Sleep_Quality',
    model_params={
        'holdout_size': 0.2,
        'use_proba': 'auto',
        'random_state': 42,
        # opcional: 'estimators': [(...), (...)] e 'final_estimator': ...
    },
    compute_importance=False,
    verbose=True
)

