from algoritmos_class import train_and_test_classifier
import pandas as pd

dataset = pd.read_csv("datasets/synthetic_coffee_health_10000.csv")

print(dataset.columns)


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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
)