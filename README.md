# Exoplanet Classification Project

Цей проєкт призначений для **уніфікації різних астрономічних баз даних (Kepler, K2, TOI)** та **класифікації екзопланет** за допомогою моделі **LightGBM**.  
Він складається з трьох основних частин:
1. **create_dataset.py** — об’єднання та уніфікація даних.
2. **gbm4.py** — навчання моделі LightGBM для класифікації.
3. **api.py** — REST API для отримання прогнозів.

---

## Встановлення та запуск

### 1. Клонування репозиторію
```bash
git clone https://github.com/yourusername/exoplanet-classifier.git
cd exoplanet-classifier
```

### 2. Створення віртуального середовища
(рекомендовано для ізоляції залежностей)
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Встановлення залежностей
```bash
pip install -r requirements.txt
```

---

## Структура проєкту

```
exoplanet-classifier/
│
├── create_dataset.py         # Уніфікація даних Kepler / K2 / TOI у спільний CSV
├── gbm4.py                   # Навчання моделі LightGBM та збереження її у pkl
├── api.py                    # FastAPI REST API для класифікації нових зразків
├── requirements.txt          # Список залежностей
└── README.md                 # Цей файл
```

---

## Кроки роботи проєкту

### 1. Підготовка датасету
Усі вихідні CSV-файли (Kepler, K2, TOI) потрібно розмістити в директорії проєкту.  
Потім виконайте:
```bash
python create_dataset.py
```

Скрипт об’єднає дані у фінальний файл `yea.csv`.

---

### 2. Навчання моделі
Після створення датасету виконайте:
```bash
python gbm4.py
```

Скрипт:
- виконує попередню обробку даних;
- відбирає найбільш корельовані ознаки;
- навчає модель LightGBM;
- будує графіки важливості ознак і втрат;
- зберігає модель у `exoplanet_lgb_model.pkl`;
- зберігає масштабувальник у `scaler.pkl`.

---

### 3. Запуск API
Після навчання можна запустити локальний REST API:
```bash
uvicorn api:app --reload
```

API буде доступне за адресою:
```
http://127.0.0.1:8000
```

---

## Використання API

### Ендпоінт `/predict`
**Метод:** `POST`  
**Опис:** приймає список числових ознак (`features`) і повертає передбачення класу.

#### Приклад запиту:
```json
{
  "features": [365.25, 12.5, 1000.3, 1.3, 0.8, 500, 0.1, 89, 0.02, 1.1, 0.05, 5778, 1.0, 1.0, 4.4, 0.02, 4.6, 0]
}
```

#### Відповідь:
```json
{
  "predicted_label": 1,
  "confidence": 0.8732
}
```

---

### Ендпоінт `/`
Перевірка, чи API працює:
```bash
GET http://127.0.0.1:8000/
```
**Відповідь:**
```json
{ "message": "Exoplanet Classifier API is running!" }
```

---

## Модель
Модель використовує **LightGBM** із налаштуванням параметрів для балансування класів і контролю перенавчання (early stopping).  
Показники виводяться у консоль після навчання:
- Accuracy  
- ROC AUC  
- Classification Report  
- Confusion Matrix  
- Feature Importance Chart

---

## Приклад повного сценарію

```bash
# 1. Створюємо об’єднаний датасет
python create_dataset.py

# 2. Навчаємо модель
python gbm4.py

# 3. Запускаємо REST API
uvicorn api:app --reload
```


## Ліцензія
Цей проєкт розповсюджується під ліцензією **MIT**.  
Ви можете вільно використовувати та модифікувати його з посиланням на автора.
