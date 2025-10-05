import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import matplotlib.pyplot as plt

# ----------------------
# 1. Зчитування датасету
# ----------------------
df = pd.read_csv('yea.csv')

# ----------------------
# 2. Попередня очистка
# ----------------------
df.fillna(0.0, inplace=True)

# Видаляємо неінформативну колонку
if 'planet_name' in df.columns:
    df.drop(columns=['planet_name'], inplace=True)

# ----------------------
# 3. Аналіз кореляцій
# ----------------------
corr = df.corr(numeric_only=True)

# Відбираємо колонки, які мають |кореляцію| > 0.05 з цільовою змінною
corr_threshold = 0.05
important_features = corr.index[abs(corr['label']) > corr_threshold].tolist()

if 'label' not in important_features:
    important_features.append('label')

df = df[important_features]

print(f"Відібрано {len(important_features)-1} ознак із кореляцією > {corr_threshold}")
print("Колонки:", [c for c in important_features if c != 'label'])

# ----------------------
# 4. Формування X і y
# ----------------------
X = df.drop(columns=['label']).values
y = df['label'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 5. Балансування класів
# ----------------------
n0 = np.sum(y_train == 0)
n1 = np.sum(y_train == 1)
scale_pos_weight = n0 / n1

# ----------------------
# 6. Навчання моделі
# ----------------------
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 10,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
    'seed': 42
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_test, label=y_test, reference=lgb_train)


evals_result = {}

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50),
        lgb.record_evaluation(evals_result)  # <-- новий спосіб
    ]
)

# ----------------------
# 7. Збереження
# ----------------------
joblib.dump(gbm, 'exoplanet_lgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved successfully!")

# ----------------------
# 8. Оцінка
# ----------------------
y_pred_prob = gbm.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"ROC AUC: {roc:.4f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ----------------------
# 9. Візуалізація важливості ознак
# ----------------------
lgb.plot_importance(gbm, max_num_features=20, importance_type='gain')
plt.title("Feature Importance (Gain)")
plt.show()

# ----------------------
# 10. Візуалізація втрат train/val
# ----------------------
train_loss = evals_result['train']['binary_logloss']
val_loss = evals_result['valid']['binary_logloss']

plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Train Loss', linewidth=2)
plt.plot(val_loss, label='Validation Loss', linewidth=2)
plt.xlabel('Boosting Rounds')
plt.ylabel('Binary Logloss')
plt.title('LightGBM Training & Validation Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
