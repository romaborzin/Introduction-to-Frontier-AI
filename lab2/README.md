# Методичка к лабораторной 2  
## «Нейронные сети в PyTorch: от MLP к CNN»

**Тема:** Обучение нейронных сетей на изображениях в PyTorch  
**Стек:** Python, PyTorch, torchvision, matplotlib, Jupyter/Colab  

---

## 1. Цель работы

Освоить практическое обучение нейронных сетей в PyTorch: от простого MLP до CNN. Понять, как работают forward pass, loss, backward pass и оптимизация. Научиться экспериментировать с гиперпараметрами и диагностировать проблемы обучения.

---

## 2. Результаты обучения

После выполнения лабораторной студент умеет:

1. Загружать и предобрабатывать датасеты изображений через `torchvision`.
2. Определять нейросеть через `nn.Module`.
3. Писать цикл обучения: forward → loss → backward → step.
4. Обучать MLP и CNN на MNIST/Fashion-MNIST.
5. Визуализировать loss и accuracy.
6. Экспериментировать с learning rate, dropout, аугментацией.
7. Сравнивать MLP и CNN.
8. Диагностировать переобучение и недообучение.

---

## 3. Пререквизиты

- Python: классы, наследование, циклы.
- NumPy: массивы, reshape.

---

## 4. Оборудование и окружение

**Вариант A (рекомендуется): [Google Colab](https://colab.research.google.com/)**
- Runtime → Change runtime type → T4 GPU (если доступен).
- На CPU тоже работает, но медленнее.
- MNIST/Fashion-MNIST загружаются автоматически.

**Вариант B: локально**
```bash
pip install torch torchvision matplotlib numpy
```

**Проверка GPU:**
```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

**Шаблон ноутбука:** преподаватель выдаёт [lab2_template.ipynb](lab2/applications/lab2_template.ipynb).

---

## 5. Выбор датасета

Студент выбирает **один** датасет:

| № | Датасет | Классов | Размер | Сложность |
|---|---|---|---|---|
| 1 | MNIST | 10 (цифры) | 70K | Лёгкий |
| 2 | Fashion-MNIST | 10 (одежда) | 70K | Средний |
| 3 | CIFAR-10 | 10 (объекты) | 60K | Сложный |

**Рекомендация:**  **Fashion-MNIST**. MNIST слишком лёгкий (99%+ достигается тривиально), CIFAR-10 требует GPU и больше времени. Fashion-MNIST — золотая середина.



---

## 6. Структура работы

Лабораторная состоит из **8 этапов**.

### Этап 1. Постановка задачи

**Что сделать:**
1. Описать датасет: что за изображения, сколько классов, размер.
2. Определить задачу: многоклассовая классификация.
3. Выбрать метрики: accuracy и confusion matrix.
4. Сформулировать цель: достичь accuracy ≥ 90% на test.

**Шаблон в ноутбуке:**
```markdown
## 1. Постановка задачи
- Датасет: Fashion-MNIST
- Тип: многоклассовая классификация (10 классов)
- Вход: изображения 28×28, grayscale
- Метрика: accuracy, confusion matrix
- Цель: accuracy ≥ 90% на test
```

---

### Этап 2. Загрузка и визуализация данных

**Что сделать:**

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Трансформации: тензор + нормализация
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # mean/std Fashion-MNIST
])

train_ds = datasets.FashionMNIST(root="./data", train=True,
                                 download=True, transform=transform)
test_ds = datasets.FashionMNIST(root="./data", train=False,
                                download=True, transform=transform)

# Разделим train на train/val
from torch.utils.data import random_split
train_size = int(0.9 * len(train_ds))
val_size = len(train_ds) - train_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**Визуализация:**
```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img, label = train_ds[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(class_names[label])
    ax.axis("off")
plt.tight_layout()
plt.show()
```

**Что записать:**
- 10 классов, баланс примерно равный.
- Изображения 28×28, grayscale.
- Нормализация: mean ≈ 0.286, std ≈ 0.353.

**Совет:** не тратить больше 20 минут. Основное — цикл обучения.

---

### Этап 3. MLP: базовая модель

**Что сделать:**

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10)
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(model)
print("Параметров:", sum(p.numel() for p in model.parameters()))
```

**Функция обучения:**
```python
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total
```

**Цикл обучения:**
```python
def fit(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:2d} | "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")
    return history

model = MLP().to(device)
history_mlp = fit(model, train_loader, val_loader, epochs=10, lr=1e-3)
```

**Ожидаемый результат:** val accuracy ~88–89% после 10 эпох.

**Визуализация:**
```python
def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout(); plt.show()

plot_history(history_mlp, "MLP")
```

**Что записать:**
- Финальные train/val метрики.
- Есть ли переобучение.
- Сколько параметров.

---

### Этап 4. Эксперименты с MLP

**Что сделать:** провести **3 эксперимента**, меняя по одному параметру.

| Эксперимент | Что меняем | Гипотеза |
|---|---|---|
| 1 | lr = 1e-1 (слишком большой) | Loss взорвётся или не сойдётся |
| 2 | lr = 1e-4 (слишком маленький) | Медленно, недообучение |
| 3 | dropout = 0.0 vs 0.5 | Без dropout — переобучение |

**Шаблон:**
```python
configs = [
    {"name": "lr=1e-1", "lr": 1e-1, "dropout": 0.2},
    {"name": "lr=1e-4", "lr": 1e-4, "dropout": 0.2},
    {"name": "dropout=0.0", "lr": 1e-3, "dropout": 0.0},
    {"name": "dropout=0.5", "lr": 1e-3, "dropout": 0.5},
]

results = {}
for cfg in configs:
    print(f"\n=== {cfg['name']} ===")
    m = MLP(dropout=cfg["dropout"]).to(device)
    h = fit(m, train_loader, val_loader, epochs=10, lr=cfg["lr"])
    results[cfg["name"]] = h
    plot_history(h, cfg["name"])
```

**Что записать в таблицу:**

| Конфиг | Train acc | Val acc | Комментарий |
|---|---|---|---|
| lr=1e-1 | ... | ... | ... |
| lr=1e-4 | ... | ... | ... |
| dropout=0.0 | ... | ... | ... |
| dropout=0.5 | ... | ... | ... |

**Ожидаемые выводы:**
- lr=1e-1: loss скачет, accuracy низкая.
- lr=1e-4: сходится медленно, accuracy ниже.
- dropout=0.0: train acc высокая, val ниже — переобучение.
- dropout=0.5: train и val ближе, но обе чуть ниже.

---

### Этап 5. CNN: свёрточная сеть

**Что сделать:** определить CNN и обучить.

```python
class CNN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)         # 7 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

cnn = CNN().to(device)
print(cnn)
print("Параметров:", sum(p.numel() for p in cnn.parameters()))

history_cnn = fit(cnn, train_loader, val_loader, epochs=10, lr=1e-3)
plot_history(history_cnn, "CNN")
```

**Ожидаемый результат:** val accuracy ~91–92%.

**Что записать:**
- Параметров у CNN больше, но accuracy выше.
- Сходится быстрее MLP.

---

### Этап 6. Сравнение MLP и CNN

**Что сделать:**

```python
comparison = pd.DataFrame({
    "MLP": [history_mlp["val_acc"][-1],
            sum(p.numel() for p in MLP().parameters())],
    "CNN": [history_cnn["val_acc"][-1],
            sum(p.numel() for p in CNN().parameters())],
}, index=["Val accuracy (10 эпох)", "Параметров"])

print(comparison)
```

**Визуализация:**
```python
plt.plot(history_mlp["val_acc"], label="MLP")
plt.plot(history_cnn["val_acc"], label="CNN")
plt.xlabel("Epoch"); plt.ylabel("Val accuracy")
plt.title("MLP vs CNN"); plt.legend(); plt.show()
```

**Что записать:**
- CNN лучше MLP на 2–3%.
- CNN использует локальность изображений.
- MLP «не видит» структуру, для него это просто 784 числа.

---

### Этап 7. Оценка на test и анализ ошибок

**Что сделать:**

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y)
        all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())

y_pred, y_true, y_proba = predict(cnn, test_loader)

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предсказано"); plt.ylabel("Реально")
plt.title("Confusion Matrix — CNN")
plt.tight_layout(); plt.show()
```

**Анализ ошибок:**
```python
# Топ-5 пар, которые модель путает
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
top_errors = np.dstack(np.unravel_index(
    np.argsort(cm_no_diag.ravel())[::-1][:5], cm.shape))[0]

for true_i, pred_i in top_errors:
    print(f"{class_names[true_i]:12s} → {class_names[pred_i]:12s}: {cm[true_i, pred_i]}")

# Визуализация ошибок
errors_idx = np.where(y_pred != y_true)[0]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, idx in enumerate(errors_idx[:10]):
    img, _ = test_ds[idx]
    ax = axes.flat[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"T:{class_names[y_true[idx]][:5]}\nP:{class_names[y_pred[idx]][:5]}",
                 fontsize=8)
    ax.axis("off")
plt.tight_layout(); plt.show()
```

**Ожидаемые выводы:**
- Основные путаницы: Shirt ↔ T-shirt, Pullover ↔ Coat, Sneaker ↔ Ankle boot.
- Это визуально похожие классы — модель ошибается там, где и человек.
- Sandal, Trouser, Bag распознаются почти идеально.

**Что записать:**
- Test accuracy.
- 3 топ-ошибки.
- 3 вывода.

---

### Этап 8. Выводы

**Что сделать:**
1. Сравнить MLP и CNN.
2. Ответить:
   - Достигли ли цели accuracy ≥ 90%?
   - Что дало наибольший прирост?
   - Где модель ошибается и почему?
3. Предложить улучшения:
   - Аугментация (повороты, сдвиги).
   - Batch Normalization.
   - Больше эпох.
   - Learning rate scheduler.
   - Ансамбль моделей.
4. Описать, как внедрили бы в продакшн.

---

## 7. Требования к сдаче

**Что сдать:**
1. Ноутбук `.ipynb` с выполненными ячейками.
2. Все графики видны в ноутбуке.
3. Текстовые выводы после каждого этапа.
4. Таблица экспериментов (этап 4).
5. Краткий отчёт (1–2 страницы).
6. Ноутбук запускается сверху вниз без ошибок.

**Чего не делать:**
- Не оставлять `# TODO`.
- Не копировать код без понимания.
- Не использовать test для подбора гиперпараметров.
- Не сдавать ноутбук с ошибками.

---

## 8. Критерии оценивания (100 баллов)

| Критерий | Баллы |
|---|---|
| Постановка задачи | 5 |
| Загрузка и визуализация данных | 10 |
| MLP: реализация и обучение | 15 |
| Эксперименты с гиперпараметрами | 20 |
| CNN: реализация и обучение | 15 |
| Сравнение MLP и CNN | 10 |
| Оценка на test и confusion matrix | 10 |
| Анализ ошибок и выводы | 10 |
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
| Забыть `optimizer.zero_grad()` | Градиенты накапливаются | Всегда в начале итерации |
| Забыть `model.train()` / `model.eval()` | Dropout/BatchNorm работают неправильно | Переключать перед epoch |
| Не переносить данные на `device` | Ошибка или медленно на CPU | `.to(device)` для x и y |
| Считать accuracy по logits вместо argmax | Ошибка размерности | `logits.argmax(1)` |
| Использовать `loss.item()` без `* batch_size` | Неверный средний loss | Умножать на размер батча |
| Забыть `@torch.no_grad()` при eval | Лишняя память | Декоратор или `with torch.no_grad()` |
| Оценивать на test каждую эпоху | Подбор по test | Использовать val |
| lr слишком большой | Loss = nan | Начать с 1e-3 |
| Не нормализовать данные | Медленная сходимость | `transforms.Normalize` |

---

## 10. Полезные ссылки

- PyTorch tutorials: https://pytorch.org/tutorials/
- torchvision: https://pytorch.org/vision/stable/
- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Neural Networks from Scratch (Karpathy): https://karpathy.ai/
- Google Colab: https://colab.research.google.com/

---

## 11. Шаблон ноутбука

```
1. Постановка задачи
   - TODO: описать датасет, метрику, цель

2. Загрузка и визуализация
   - TODO: transforms, DataLoader
   - TODO: разделить train/val
   - TODO: визуализировать примеры

3. MLP
   - TODO: определить nn.Module
   - TODO: train/eval функции
   - TODO: обучить 10 эпох
   - TODO: графики loss/accuracy

4. Эксперименты
   - TODO: lr=1e-1, lr=1e-4, dropout=0.0, dropout=0.5
   - TODO: таблица результатов

5. CNN
   - TODO: определить CNN
   - TODO: обучить
   - TODO: графики

6. Сравнение MLP vs CNN
   - TODO: таблица и график

7. Test + анализ ошибок
   - TODO: classification_report
   - TODO: confusion matrix
   - TODO: визуализация ошибок
   - TODO: 3 вывода

8. Выводы
   - TODO: итог, ограничения, улучшения
```

---


## 12. Дополнительное задание (+5 баллов)

**Аугментация данных:**
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
```
Обучить CNN с аугментацией и сравнить с базовым. Ожидаемый прирост: +1–2% val accuracy.

**Batch Normalization:**
Добавить `nn.BatchNorm2d` после каждой свёртки. Сравнить скорость сходимости.

**Learning rate scheduler:**
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
```
Показать, как меняется loss при снижении lr.

**Визуализация фильтров:**
```python
w = cnn.features[0].weight.data.cpu()
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(w[i, 0], cmap="gray")
    ax.axis("off")
plt.suptitle("Фильтры первого слоя CNN")
plt.show()
```
Показать, что первый слой учит края и простые текстуры.

---

## 13. Приложение: [utils.py](lab2/applications/utils.py)

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer,
                                      criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1:2d} | "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f}")
    return history


def plot_history(history, title="Model"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y)
        all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_probs).numpy())


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Предсказано"); plt.ylabel("Реально")
    plt.title("Confusion Matrix")
    plt.tight_layout(); plt.show()
    return cm


def set_seed(seed=42):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed = {seed}")
```

---

## 14. Приложение: [requirements.txt](lab2/applications/requirements.txt)

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
pandas>=2.0.0
jupyter>=1.0.0
ipykernel>=6.25.0
```

---
