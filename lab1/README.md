# Методичка к лабораторной 1  
## «End-to-end ML на табличных данных»

**Тема:** Полный пайплайн машинного обучения на табличных данных  
**Стек:** Python, pandas, scikit-learn, matplotlib/seaborn, Jupyter/Colab

---

## 1. Цель работы

Освоить полный цикл ML на табличных данных: от постановки задачи и разведочного анализа до обучения, оценки и анализа ошибок модели. Научиться делать обоснованные выводы и честно оценивать качество.

---

## 2. Результаты обучения

После выполнения лабораторной студент умеет:

1. Формулировать задачу ML и выбирать метрику.
2. Проводить EDA и готовить данные.
3. Обучать модели классификации и регрессии.
4. Правильно делить данные на train/val/test.
5. Сравнивать модели по метрикам.
6. Диагностировать переобучение и недообучение.
7. Анализировать ошибки и делать выводы.

---

## 3. Пререквизиты

- Python: функции, списки, словари, циклы.
- pandas: `read_csv`, `head`, `describe`, `groupby`.
- matplotlib/seaborn: базовые графики.

---

## 4. Оборудование и окружение

**Вариант A (рекомендуется): [Google Colab](https://colab.research.google.com/)**
- Не требует установки.
- GPU не нужен.
- Достаточно аккаунта Google.

**Вариант B: локально**
```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
```

**Шаблон ноутбука:** преподаватель выдаёт [lab1_template.ipynb](lab1/applications/lab1_template.ipynb) со структурой и `# TODO`-ячейками.

---

## 5. Выбор датасета

Студент выбирает **один** датасет из списка или предлагает свой (согласовать с преподавателем).

| № | Датасет | Задача | Ссылка |
|---|---|---|---|
| 1 | Titanic | Классификация: выжил/нет | Kaggle / seaborn |
| 2 | Telco Customer Churn | Классификация: уйдёт/нет | Kaggle |
| 3 | California Housing | Регрессия: цена дома | sklearn.datasets |
| 4 | Heart Disease UCI | Классификация: болезнь | UCI |
| 5 | Adult Income | Классификация: доход >50K | UCI |

**Требования к датасету:**
- Минимум 500 строк.
- Минимум 5 признаков.
- Есть числовые и категориальные признаки.
- Есть целевая переменная.

---

## 6. Структура работы

Лабораторная состоит из **7 этапов**. Каждый этап — отдельная секция ноутбука.

### Этап 1. Постановка задачи

**Что сделать:**
1. Описать датасет: откуда, сколько строк, что означает каждая колонка.
2. Определить тип задачи: классификация или регрессия.
3. Определить target.
4. Выбрать метрику и обосновать выбор.
5. Сформулировать бизнес-цель: зачем вообще эта модель.

**Шаблон в ноутбуке:**
```markdown
## 1. Постановка задачи
- Датасет: ...
- Тип задачи: ...
- Target: ...
- Метрика: ...
- Почему эта метрика: ...
- Бизнес-цель: ...
```

**Пример:**
- Датасет: Telco Churn.
- Тип: бинарная классификация.
- Target: `Churn` (Yes/No).
- Метрика: F1 и recall (важнее не пропустить уходящего клиента).
- Бизнес-цель: удержать клиентов через таргетированные предложения.

---

### Этап 2. EDA — разведочный анализ

**Что сделать:**
1. Загрузить данные: `df = pd.read_csv(...)`.
2. Посмотреть: `df.head()`, `df.info()`, `df.describe()`.
3. Проверить пропуски: `df.isnull().sum()`.
4. Посмотреть распределение target: `df['target'].value_counts()`.
5. Построить графики:
   - Гистограммы числовых признаков.
   - Bar plot категориальных.
   - Correlation heatmap.
   - Boxplot target vs числовые признаки.
6. Записать **минимум 3 наблюдения** текстом.

**Пример наблюдений:**
- «Классы несбалансированы: 73% / 27%.»
- «Признак `TotalCharges` содержит пропуски — нужно обработать.»
- «`MonthlyCharges` сильно коррелирует с target.»

**Совет:** не тратить больше 30 минут. EDA — не самоцель, а подготовка.

---

### Этап 3. Подготовка данных

**Что сделать:**
1. Обработать пропуски:
   - Числовые: медиана или среднее.
   - Категориальные: мода или отдельная категория «Unknown».
2. Закодировать категориальные признаки:
   - One-Hot Encoding для номинальных.
   - Label Encoding для порядковых.
3. Масштабировать числовые признаки (для линейных моделей и kNN):
   - `StandardScaler` или `MinMaxScaler`.
4. Разделить данные:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```
5. Внутри train — выделить validation через кросс-валидацию.

**Важно:**
- `fit` scaler только на train, `transform` на test — иначе утечка данных.
- Использовать `Pipeline`, чтобы не ошибиться:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LogisticRegression

  pipe = Pipeline([
      ('scaler', StandardScaler()),
      ('model', LogisticRegression())
  ])
  ```

**Частая ошибка:** масштабировать весь датасет до split. Это data leakage.

---

### Этап 4. Обучение моделей

**Что сделать:**
Обучить **минимум 3 модели**:

| Модель | Зачем |
|---|---|
| Логистическая регрессия (или линейная) | Baseline, интерпретируемость |
| Decision Tree | Нелинейность, интерпретируемость |
| Random Forest или Gradient Boosting | Качество на табличных данных |

**Шаги:**
1. Обучить каждую модель на train.
2. Оценить через кросс-валидацию на train:
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1')
   print(scores.mean(), scores.std())
   ```
3. Записать результаты в таблицу.

**Шаблон таблицы:**

| Модель | CV F1 (mean ± std) | Время обучения |
|---|---|---|
| LogReg | ... | ... |
| Tree | ... | ... |
| RF | ... | ... |

**Совет:** не гнаться за идеалом. Цель — сравнить, а не победить в соревновании.

---

### Этап 5. Оценка на test

**Что сделать:**
1. Выбрать лучшую модель по CV.
2. Обучить на всём train.
3. Предсказать на test.
4. Посчитать метрики:
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   print(classification_report(y_test, y_pred))
   print(confusion_matrix(y_test, y_pred))
   ```
5. Для регрессии:
   ```python
   from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
   ```

**Важно:**
- Test трогаем **один раз**. Если подбирать модель по test — это уже переобучение.
- Сравнить test-метрику с CV: если сильно ниже — возможна проблема.

**Что записать:**
- Итоговые метрики.
- Матрицу ошибок.
- Вывод: модель переобучена, недообучена или адекватна.

---

### Этап 6. Анализ ошибок

**Что сделать:**
1. Найти примеры, где модель ошиблась:
   ```python
   errors = X_test[y_test != y_pred]
   ```
2. Посмотреть на FP и FN отдельно.
3. Построить графики:
   - Confusion matrix heatmap.
   - ROC-кривая (для классификации).
   - Residual plot (для регрессии).
4. Ответить на вопросы:
   - Какие классы модель путает чаще?
   - Есть ли паттерн в ошибках (например, молодые клиенты)?
   - Какие признаки важны? `model.feature_importances_` или коэффициенты.
5. Записать **минимум 3 вывода**.

**Пример вывода:**
- «Модель чаще ошибается на клиентах с маленьким сроком контракта.»
- «Recall по классу "ушёл" = 0.55 — много пропусков.»
- «Наиболее важные признаки: `Contract`, `Tenure`, `MonthlyCharges`.»

---

### Этап 7. Выводы и рекомендации

**Что сделать:**
1. Сформулировать итог: что получилось, что нет.
2. Ответить:
   - Достигли ли цели по метрике?
   - Какая модель лучшая и почему?
   - Какие ограничения у решения?
3. Предложить улучшения:
   - Больше данных.
   - Feature engineering.
   - Другие модели (XGBoost, CatBoost).
   - Балансировка классов (SMOTE, class_weight).
   - Подбор гиперпараметров (GridSearchCV, Optuna).
4. Описать, как внедрили бы модель в продакшн.

---

## 7. Требования к сдаче

**Что сдать:**
1. Ноутбук `.ipynb` с выполненными ячейками.
2. Все графики должны быть видны в ноутбуке.
3. Текстовые выводы после каждого этапа.
4. Краткий отчёт (1–2 страницы) или README:
   - Постановка задачи.
   - Что сделано.
   - Результаты.
   - Выводы.
5. Ноутбук должен запускаться сверху вниз без ошибок.

**Чего не делать:**
- Не оставлять `# TODO`.
- Не копировать код без понимания.
- Не использовать test для подбора модели.
- Не сдавать ноутбук с ошибками выполнения.

---

## 8. Критерии оценивания (100 баллов)

| Критерий | Баллы |
|---|---|
| Постановка задачи и выбор метрики | 10 |
| EDA: графики и наблюдения | 15 |
| Подготовка данных: пропуски, кодирование, масштабирование | 15 |
| Обучение 3+ моделей и кросс-валидация | 15 |
| Оценка на test и матрица ошибок | 15 |
| Анализ ошибок и выводы | 15 |
| Качество выводов и рекомендаций | 10 |
| Оформление и воспроизводимость | 5 |

**Шкала:**
- 90–100: отлично.
- 75–89: хорошо.
- 60–74: удовлетворительно.
- <60: требуется доработка.

---

## 9. Типичные ошибки

| Ошибка | Последствие | Как избежать |
|---|---|---|
| Масштабирование до split | Data leakage | `Pipeline` или fit только на train |
| Подбор модели по test | Переобучение на test | Test трогать один раз |
| Игнорирование дисбаланса | Accuracy обманчива | Смотреть precision/recall/F1 |
| Заполнение пропусков средним по всему датасету | Утечка | Заполнять по train |
| Отсутствие `random_state` | Невоспроизводимость | Всегда указывать |
| Нет текстовых выводов | Непонятно, что студент понял | Писать после каждого этапа |
| Слишком много моделей без анализа | Время потрачено зря | 3–4 модели достаточно |

---

## 10. Полезные ссылки

- scikit-learn: https://scikit-learn.org/stable/
- pandas: https://pandas.pydata.org/docs/
- seaborn: https://seaborn.pydata.org/
- Kaggle Datasets: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml/
- Google Colab: https://colab.research.google.com/

---

## 11. Шаблон ноутбука

Файл `lab1_template.ipynb` со структурой:

```
1. Постановка задачи
   - TODO: описать датасет
   - TODO: тип задачи, target, метрика

2. EDA
   - TODO: загрузить данные
   - TODO: пропуски, распределения
   - TODO: графики
   - TODO: 3 наблюдения

3. Подготовка данных
   - TODO: обработка пропусков
   - TODO: кодирование
   - TODO: масштабирование
   - TODO: train/test split

4. Обучение моделей
   - TODO: 3 модели
   - TODO: кросс-валидация
   - TODO: таблица результатов

5. Оценка на test
   - TODO: метрики
   - TODO: confusion matrix / residual plot

6. Анализ ошибок
   - TODO: FP/FN
   - TODO: feature importance
   - TODO: 3 вывода

7. Выводы
   - TODO: итог
   - TODO: ограничения
   - TODO: улучшения
```

---


## 12. Приложение: вспомогательные функции ([utils.py](lab1/applications/utils.py))


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)


def classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(metrics, title="Метрики"):
    print(f"=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Предсказано")
    plt.ylabel("Реально")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return cm


def plot_roc_curve(y_true, y_proba, title="ROC-кривая"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Модель не имеет feature_importances_ или coef_")
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    sns.barplot(data=df, x="importance", y="feature")
    plt.title("Важность признаков")
    plt.tight_layout()
    plt.show()
    return df


def missing_report(df):
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        print("Пропусков нет.")
        return pd.DataFrame()
    report = pd.DataFrame({
        "missing": miss,
        "percent": (miss / len(df) * 100).round(2)
    })
    print(report)
    return report


def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed = {seed}")
```

---

## 13. Приложение: [requirements.txt](lab1/applications/requirements.txt)

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.25.0
xgboost>=2.0.0
lightgbm>=4.0.0
```
