# Методичка к лабораторной 3  
## «Трансформеры и LLM: fine-tuning и применение в NLP»

**Курс:** Введение в фронтирные ИИ  
**Тема:** Работа с предобученными трансформерами через Hugging Face   
**Стек:** Python, PyTorch, Hugging Face Transformers, Datasets, Evaluate, Jupyter/Colab  

---

## 1. Цель работы

Освоить практическую работу с предобученными трансформерами: токенизацию, fine-tuning BERT для классификации текста, визуализацию attention, знакомство с GPT-2 для генерации. Понять, как устроен современный NLP-пайплайн и почему pretraining + fine-tuning — стандарт индустрии.

---

## 2. Результаты обучения

После выполнения лабораторной студент умеет:

1. Загружать предобученные модели и токенизаторы через Hugging Face.
2. Разбираться в токенизации: subword, BPE, специальные токены.
3. Дообучать BERT для классификации текста.
4. Оценивать модель: accuracy, F1, confusion matrix.
5. Визуализировать attention.
6. Генерировать текст через GPT-2 и управлять генерацией.
7. Сравнивать модели: zero-shot vs fine-tuned.
8. Понимать, когда fine-tuning, а когда prompting/RAG.

---

## 3. Пререквизиты

- Python: классы, циклы, списки, словари.
- PyTorch: тензоры, `nn.Module`, DataLoader.
- Основы NLP: токены, эмбеддинги.

---

## 4. Оборудование и окружение

**Вариант A (рекомендуется): [Google Colab](https://colab.research.google.com/)**
- Runtime → Change runtime type → T4 GPU.
- BERT-base fine-tuning на 10K примеров: ~2–3 минуты на T4.
- На CPU: ~30–40 минут — терпимо, но лучше GPU.

**Вариант B: локально**
```bash
pip install torch transformers datasets evaluate accelerate scikit-learn
pip install bertviz matplotlib seaborn
```

**Проверка окружения:**
```python
import torch
import transformers
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
```

**Шаблон ноутбука:** преподаватель выдаёт [lab3_template.ipynb](lab3/applications/lab3_template.ipynb).

---

## 5. Выбор задачи и датасета

Студент выбирает **одну** задачу:

| № | Задача | Датасет | Модель | Сложность |
|---|---|---|---|---|
| 1 | Классификация тональности | IMDb / Rotten Tomatoes | BERT-base | Средний |
| 2 | Классификация новостей | AG News | BERT-base | Средний |
| 3 | Определение токсичности | Jigsaw Toxic Comments | BERT-base | Сложный |
| 4 | Русская тональность | RuSentiment / Kinopoisk | RuBERT | Средний |
| 5 | NLI (логическое следствие) | SNLI (подвыборка) | BERT-base | Сложный |

**Рекомендация:** **IMDb** (английская тональность) или **RuSentiment** (русская). Простая бинарная задача, понятные метрики, быстрый fine-tuning.

**Требования:**
- Минимум 5 000 примеров в train.
- Бинарная или многоклассовая классификация.
- Текст на английском или русском.

**Совет:** если GPU нет и Colab тормозит — взять подвыборку 5 000 train / 1 000 test.

---

## 6. Структура работы

Лабораторная состоит из **8 этапов**.

### Этап 1. Постановка задачи (5 мин)

**Что сделать:**
1. Описать датасет: домен, размер, классы.
2. Определить задачу: классификация.
3. Выбрать метрики: accuracy и F1-macro (при дисбалансе).
4. Сформулировать цель: F1 ≥ 0.85.
5. Выбрать базовую модель: `bert-base-uncased` или `DeepPavlov/rubert-base-cased`.

**Шаблон:**
```markdown
## 1. Постановка задачи
- Датасет: IMDb (50K отзывов, 2 класса)
- Тип: бинарная классификация тональности
- Метрика: accuracy + F1-macro
- Цель: F1 ≥ 0.90
- Модель: bert-base-uncased (110M параметров)
- Baseline: zero-shot через pipeline
```

---

### Этап 2. Знакомство с токенизацией (20 мин)

**Что сделать:**

```python
from transformers import AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Примеры токенизации
texts = [
    "This movie was absolutely fantastic!",
    "I hated every minute of it.",
    "Машинное обучение — это интересно.",
    "unbelievably great",
    "tokenization123"
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    print(f"Текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Длина: {len(tokens)} токенов")
    print("-" * 60)
```

**Вывод (пример):**

```
Текст: This movie was absolutely fantastic!
Токены: ['this', 'movie', 'was', 'absolutely', 'fantastic', '!']
IDs: [101, 2023, 3185, 2004, 7078, 13237, 999, 102]
Длина: 6 токенов
------------------------------------------------------------
Текст: Машинное обучение — это интересно.
Токены: ['ма', '##шин', '##ное', 'обу', '##чение', '—', 'это', 'инте', '##ресно', '.']
IDs: [101, 4845, 4574, 2680, 12606, 17445, 117, 2756, 13412, 14016, 1012, 102]
Длина: 10 токенов
------------------------------------------------------------
Текст: unbelievably great
Токены: ['unbelie', '##vable', '##ly', 'great']
IDs: [101, 4538, 11100, 2987, 2307, 102]
Длина: 4 токенов
```

**Анализ:**

```python
# Специальные токены
print("CLS:", tokenizer.cls_token, tokenizer.cls_token_id)
print("SEP:", tokenizer.sep_token, tokenizer.sep_token_id)
print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
print("UNK:", tokenizer.unk_token, tokenizer.unk_token_id)
print("Vocab size:", tokenizer.vocab_size)

# Сравнение: русский vs английский
en = "Machine learning is fascinating."
ru = "Машинное обучение — это интересно."
print(f"EN tokens: {len(tokenizer.tokenize(en))}")
print(f"RU tokens: {len(tokenizer.tokenize(ru))}")
```

**Вывод:**

```
CLS: [CLS] 101
SEP: [SEP] 102
PAD: [PAD] 0
UNK: [UNK] 100
Vocab size: 30522

EN tokens: 6
RU tokens: 10
```

### Ячейка (markdown)

```markdown
### Наблюдения по токенизации

1. **Subword (WordPiece).** Редкие слова разбиваются на куски:
   `unbelievably` → `unbelie` + `##vable` + `##ly`.
   `##` означает «продолжение слова».

2. **Специальные токены:**
   - `[CLS]` — в начало, его эмбеддинг используется для классификации.
   - `[SEP]` — разделитель предложений.
   - `[PAD]` — для выравнивания длины батча.
   - `[UNK]` — неизвестные токены.

3. **Русский токенизируется хуже.** Одно и то же по смыслу
   предложение: EN — 6 токенов, RU — 10 токенов. Причина:
   BERT-base обучен в основном на английском.

4. **Вывод:** для русских задач нужно брать `rubert-base-cased`,
   `DeepPavlov/rubert-base-cased` или `ai-forever/ruBert-base`.
   Это уменьшит число токенов и улучшит качество.
```

---

### Этап 3. Zero-shot baseline через pipeline

**Что сделать:**

```python
from transformers import pipeline

# Zero-shot classification
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      device=0 if torch.cuda.is_available() else -1)

texts = [
    "This movie was absolutely fantastic!",
    "I hated every minute of it.",
    "It was okay, nothing special.",
    "A masterpiece of modern cinema."
]

for text in texts:
    result = classifier(text)[0]
    print(f"{text}")
    print(f"  → {result['label']}: {result['score']:.4f}")
```

**Вывод:**

```
This movie was absolutely fantastic!
  → POSITIVE: 0.9998
I hated every minute of it.
  → NEGATIVE: 0.9996
It was okay, nothing special.
  → POSITIVE: 0.9871  ← ошибка, но пограничный случай
A masterpiece of modern cinema.
  → POSITIVE: 0.9997
```

### Ячейка (markdown)

```markdown
### Zero-shot baseline

`pipeline` даёт готовую модель без обучения. Это **baseline** —
сравним с ним наш fine-tuned BERT.

**Плюсы zero-shot:**
- Не нужно обучать.
- Работает «из коробки».

**Минусы:**
- Модель обучена на другом домене (IMDb → SST-2).
- Не адаптирована к специфике нашей задачи.
- Точность на нашем датасете будет ниже.

**Цель:** показать, что fine-tuning даёт +5–10% F1.
```

---

### Этап 4. Загрузка и подготовка данных

**Что сделать:**

```python
from datasets import load_dataset

# IMDb
dataset = load_dataset("imdb")
print(dataset)

# Подвыборка для скорости (если нужно)
train_ds = dataset["train"].shuffle(seed=42).select(range(5000))
test_ds = dataset["test"].shuffle(seed=42).select(range(1000))

print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
print("\nПример:")
print(train_ds[0])
```

**Вывод:**

```
DatasetDict({
    train: Dataset({features: ['text', 'label'], num_rows: 25000})
    test: Dataset({features: ['text', 'label'], num_rows: 25000})
    unsupervised: Dataset({features: ['text', 'label'], num_rows: 50000})
})
Train: 5000, Test: 1000

Пример:
{'text': 'I rented this movie...', 'label': 1}
```

**Токенизация датасета:**

```python
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=256)

train_tok = train_ds.map(tokenize_fn, batched=True)
test_tok = test_ds.map(tokenize_fn, batched=True)

train_tok = train_tok.remove_columns(["text"])
test_tok = test_tok.remove_columns(["text"])
train_tok.set_format("torch")
test_tok.set_format("torch")

print(train_tok[0].keys())
print("input_ids shape:", train_tok[0]["input_ids"].shape)
```

**Вывод:**

```
dict_keys(['label', 'input_ids', 'token_type_ids', 'attention_mask'])
input_ids shape: torch.Size([256])
```

**Разделение train/val:**

```python
split = train_tok.train_test_split(test_size=0.1, seed=42)
train_tok = split["train"]
val_tok = split["test"]
print(f"Train: {len(train_tok)}, Val: {len(val_tok)}")
```

### Ячейка (markdown)

```markdown
### Решения по данным

- **max_length=256** — компромисс: IMDb отзывы бывают длинными,
  но 256 покрывает большинство. Для 512+ — медленнее.
- **Padding="max_length"** — просто, но неэффективно.
  В продакшне лучше `DataCollatorWithPadding` для динамического паддинга.
- **Train/Val split 90/10** — стандарт.
- **Test не трогаем до финала.**
```

---

### Этап 5. Fine-tuning BERT

**Что сделать:**

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Модель
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
).to(device)

# Метрики
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# Аргументы обучения
args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    report_to="none",
    seed=42
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    compute_metrics=compute_metrics
)

# Обучение
trainer.train()
```

**Вывод (пример):**

```
Epoch 1/3 | loss 0.3521 | val_acc 0.8920 | val_f1 0.8918
Epoch 2/3 | loss 0.1843 | val_acc 0.9210 | val_f1 0.9208
Epoch 3/3 | loss 0.1236 | val_acc 0.9340 | val_f1 0.9339

Training time: 178 s
```

**График (описание):** loss падает с 0.35 до 0.12; val_f1 растёт с 0.89 до 0.93. Признаков переобучения нет — 3 эпохи для BERT на 5K примеров — норма.

### Ячейка (markdown)

```markdown
### Fine-tuning: ключевые гиперпараметры

| Параметр | Значение | Почему |
|---|---|---|
| learning_rate | 2e-5 | Стандарт для BERT. Больше — разрушит веса |
| epochs | 3 | BERT переобучается после 3–5 эпох |
| batch_size | 16 | Ограничение памяти GPU |
| warmup_ratio | 0.1 | Плавный старт lr, стабилизирует обучение |
| weight_decay | 0.01 | Лёгкая регуляризация |
| max_length | 256 | Компромисс скорость/качество |

**Почему lr такой маленький?**
BERT уже предобучен. Большой lr «сломает» выученные веса.
Мы лишь слегка адаптируем модель под задачу.
```

---

### Этап 6. Оценка на test

**Что сделать:**

```python
from sklearn.metrics import classification_report, confusion_matrix

# Предсказания
preds = trainer.predict(test_tok)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = preds.label_ids

print(classification_report(y_true, y_pred,
                            target_names=["Negative", "Positive"]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Предсказано")
plt.ylabel("Реально")
plt.title("BERT fine-tuned — Confusion Matrix (test)")
plt.tight_layout()
plt.show()
```

**Вывод:**

```
              precision    recall  f1-score   support

    Negative       0.93      0.94      0.93       500
    Positive       0.94      0.93      0.93       500

    accuracy                           0.935      1000
   macro avg       0.935     0.935     0.935      1000
weighted avg       0.935     0.935     0.935      1000
```

### Ячейка (markdown)

```markdown
### Сравнение с baseline

| Модель | Accuracy | F1-macro |
|---|---|---|
| Zero-shot (DistilBERT SST-2) | ~0.85 | ~0.85 |
| **Fine-tuned BERT-base** | **0.935** | **0.935** |

**Fine-tuning дал +8.5% F1** относительно zero-shot.
Это типичный результат для доменной адаптации.
```

---

### Этап 7. Анализ ошибок и attention

**Что сделать:**

**7.1. Примеры ошибок**

```python
# Найдём ошибки
errors = test_ds.select(np.where(y_pred != y_true)[0][:5].tolist())
for i, ex in enumerate(errors):
    pred = y_pred[np.where(y_pred != y_true)[0][i]]
    true = y_true[np.where(y_pred != y_true)[0][i]]
    print(f"--- Ошибка {i+1} ---")
    print(f"Текст: {ex['text'][:300]}...")
    print(f"Истина: {'Positive' if true == 1 else 'Negative'}")
    print(f"Предсказано: {'Positive' if pred == 1 else 'Negative'}")
    print()
```

**Вывод (пример):**

```
--- Ошибка 1 ---
Текст: This movie is so bad it's almost good. The acting is
terrible, but somehow I couldn't stop watching...
Истина: Positive
Предсказано: Negative
(Сарказм и смешанные сигналы — сложно для модели)

--- Ошибка 2 ---
Текст: I expected more from this director. Not his best,
but still worth watching.
Истина: Positive
Предсказано: Negative
(Смешанная тональность, «not his best» перевесило)
```

**7.2. Анализ по длине текста**

```python
lengths = [len(t.split()) for t in test_ds["text"]]
df = pd.DataFrame({
    "length": lengths,
    "correct": (y_pred == y_true)
})
df["length_bin"] = pd.cut(df["length"],
                          bins=[0, 50, 100, 200, 500, 10000],
                          labels=["0-50", "50-100", "100-200",
                                  "200-500", "500+"])
acc_by_len = df.groupby("length_bin")["correct"].mean()
print(acc_by_len)
```

**Вывод:**

```
length_bin
0-50       0.912
50-100     0.941
100-200    0.938
200-500    0.930
500+       0.918
```

**7.3. Визуализация attention**

```python
from bertviz import head_view

# Возьмём одну фразу
sentence = "This movie was not good at all, but the acting was brilliant."
inputs = tokenizer(sentence, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
attentions = outputs.attentions  # tuple of (num_layers, ...)

# Визуализация (в Colab работает)
head_view(attentions, tokens)
```

**График (описание):** интерактивный HTML с heatmap attention. Видно, что на слове `good` модель обращает внимание на `not` — важный сигнал для отрицания.

**7.4. Feature importance через occlusion**

```python
def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=-1)[0, 1].item()

sentence = "The plot was predictable but the acting saved the film."
words = sentence.split()
base_prob = predict_proba(sentence)
print(f"Базовая вероятность Positive: {base_prob:.3f}\n")

for i in range(len(words)):
    modified = " ".join(words[:i] + words[i+1:])
    prob = predict_proba(modified)
    delta = base_prob - prob
    marker = "***" if abs(delta) > 0.1 else ""
    print(f"Убрать '{words[i]:15s}' → Δ = {delta:+.3f} {marker}")
```

**Вывод:**

```
Базовая вероятность Positive: 0.612

Убрать 'The            ' → Δ = +0.002
Убрать 'plot           ' → Δ = +0.031
Убрать 'was            ' → Δ = +0.003
Убрать 'predictable    ' → Δ = +0.127 ***
Убрать 'but            ' → Δ = -0.084
Убрать 'the            ' → Δ = +0.005
Убрать 'acting         ' → Δ = -0.183 ***
Убрать 'saved          ' → Δ = -0.211 ***
Убрать 'the            ' → Δ = +0.002
Убрать 'film.          ' → Δ = +0.014
```

### Ячейка (markdown)

```markdown
### Анализ ошибок — выводы

1. **Основные ошибки — сарказм и смешанная тональность.**
   «This movie is so bad it's almost good» — модель видит
   `bad`, `terrible` и не понимает, что это ирония.

2. **Длина текста почти не влияет** на accuracy:
   91–94% в диапазоне от 0 до 500+ слов. Truncation на 256
   не критична для IMDb.

3. **Attention показывает:** на слове `good` модель смотрит
   на `not` — значит, отрицание улавливается. Это подтверждает,
   что BERT работает не «по ключевым словам», а учитывает контекст.

4. **Occlusion importance:** убрать `saved` и `acting` сильнее
   всего снижает вероятность Positive. Убрать `predictable` —
   повышает. Это семантически осмысленно.

5. **Ошибки не случайны:** они кластеризуются вокруг
   языковой неоднозначности, а не вокруг длины или домена.
```

---

### Этап 8. GPT-2: генерация текста

**Что сделать:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

prompt = "Machine learning is"
inputs = gpt_tokenizer(prompt, return_tensors="pt").to(device)

# Разные стратегии генерации
outputs_greedy = gpt_model.generate(
    **inputs, max_new_tokens=40, do_sample=False,
    pad_token_id=gpt_tokenizer.eos_token_id
)
print("GREEDY:")
print(gpt_tokenizer.decode(outputs_greedy[0], skip_special_tokens=True))
print()

outputs_sample = gpt_model.generate(
    **inputs, max_new_tokens=40, do_sample=True,
    temperature=0.7, top_k=50, top_p=0.95,
    pad_token_id=gpt_tokenizer.eos_token_id
)
print("SAMPLING (T=0.7, top-k=50, top-p=0.95):")
print(gpt_tokenizer.decode(outputs_sample[0], skip_special_tokens=True))
print()

outputs_temp = gpt_model.generate(
    **inputs, max_new_tokens=40, do_sample=True,
    temperature=1.5, top_k=0,
    pad_token_id=gpt_tokenizer.eos_token_id
)
print("TEMPERATURE=1.5:")
print(gpt_tokenizer.decode(outputs_temp[0], skip_special_tokens=True))
```

**Вывод (пример):**

```
GREEDY:
Machine learning is a field of computer science that gives
computers the ability to learn without being explicitly programmed.

SAMPLING (T=0.7, top-k=50, top-p=0.95):
Machine learning is the study of algorithms that can learn from
data and make predictions. It's a subfield of AI with applications
in everything from healthcare to finance.

TEMPERATURE=1.5:
Machine learning is a strange and powerful thing, sometimes
it plays chess and other days it plays checkers, but what does
it really mean to be a machine that learns?
```

### Ячейка (markdown)

```markdown
### Стратегии генерации

| Стратегия | Что делает | Плюсы | Минусы |
|---|---|---|---|
| Greedy | Берёт самый вероятный токен | Стабильно | Скучно, повторы |
| Sampling (T=0.7) | Сэмплирование с температурой | Разнообразно | Может отклоняться |
| T=0.1 | Почти greedy | Точнее | Однообразно |
| T=1.5 | Сильно случайно | Креативно | Бессвязно |
| Top-k | Из top-k токенов | Контроль | Топ-k фиксирован |
| Top-p | Ядро вероятностей | Адаптивно | Сложнее настраивать |

**Вывод:** для фактов — T=0.2–0.5, для творчества — T=0.8–1.0.
```

---

## 7. Требования к сдаче

**Что сдать:**
1. Ноутбук `.ipynb` с выполненными ячейками.
2. Графики: confusion matrix, кривые обучения.
3. Attention-визуализация (скриншот или HTML).
4. Текстовые выводы после каждого этапа.
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
| Постановка задачи и выбор модели | 5 |
| Токенизация: примеры и анализ | 10 |
| Zero-shot baseline | 10 |
| Подготовка данных | 10 |
| Fine-tuning BERT | 20 |
| Оценка на test | 10 |
| Анализ ошибок и attention | 15 |
| GPT-2: генерация и анализ стратегий | 10 |
| Выводы и оформление | 10 |

**Шкала:**
- 90–100: отлично.
- 75–89: хорошо.
- 60–74: удовлетворительно.
- <60: требуется доработка.

---

## 9. Типичные ошибки

| Ошибка | Последствие | Как избежать |
|---|---|---|
| lr = 1e-3 (как для CNN) | Разрушение весов BERT | Использовать 2e-5 |
| Слишком много эпох (>5) | Переобучение | 3 эпохи для начала |
| Забыть `truncation=True` | Ошибка при длинных текстах | Всегда включать |
| Не использовать `attention_mask` | Модель смотрит на padding | Передавать mask |
| `padding="max_length"` вместо collator | Медленно | `DataCollatorWithPadding` |
| Оценка на test каждую эпоху | Переобучение на test | Использовать val |
| Прямое сравнение logits двух моделей | Разные шкалы | Использовать softmax |
| GPT-2 генерация без `pad_token_id` | Warning / ошибка | Указывать явно |
| Слишком длинная генерация | Мусор в конце | `max_new_tokens=40–100` |
| Игнорирование seed | Невоспроизводимость | `seed=42` везде |

---

## 10. Полезные ссылки

- Hugging Face Course: https://huggingface.co/learn/nlp-course
- Transformers docs: https://huggingface.co/docs/transformers/
- Datasets: https://huggingface.co/docs/datasets/
- BertViz: https://github.com/jessevig/bertviz
- The Illustrated BERT: https://jalammar.github.io/illustrated-bert/
- The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2/

---

## 11. Шаблон ноутбука

```
1. Постановка задачи
   - TODO: датасет, метрика, цель, модель

2. Токенизация
   - TODO: загрузить токенизатор
   - TODO: примеры на EN/RU
   - TODO: специальные токены
   - TODO: выводы

3. Zero-shot baseline
   - TODO: pipeline
   - TODO: 5–10 примеров
   - TODO: оценка на test

4. Подготовка данных
   - TODO: load_dataset
   - TODO: tokenize_fn
   - TODO: train/val split

5. Fine-tuning
   - TODO: AutoModelForSequenceClassification
   - TODO: TrainingArguments
   - TODO: Trainer, train()

6. Оценка на test
   - TODO: classification_report
   - TODO: confusion matrix
   - TODO: сравнение с baseline

7. Анализ ошибок и attention
   - TODO: 5 ошибок
   - TODO: анализ по длине
   - TODO: attention-визуализация
   - TODO: occlusion importance
   - TODO: 3–5 выводов

8. GPT-2 генерация
   - TODO: greedy, sampling, T=1.5
   - TODO: анализ стратегий

9. Выводы
   - TODO: итог, ограничения, улучшения
```

---



## 12. Дополнительное задание (+5 баллов)

**12.1. Сравнение моделей**
Попробовать `distilbert-base-uncased`, `roberta-base`, `albert-base-v2`. Сравнить: качество, скорость, размер.

**12.2. LoRA fine-tuning**
Использовать `peft` для fine-tuning с 0.5% обучаемых параметров. Сравнить с full fine-tuning по F1 и памяти.

```python
from peft import LoraConfig
