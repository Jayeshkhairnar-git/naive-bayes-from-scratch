# Naive Bayes Classifier — Built from Scratch

A fully custom implementation of a **Naive Bayes classifier** in pure Python — no scikit-learn, no shortcuts. Supports both **continuous** (Gaussian) and **categorical** (Laplace-smoothed) features, making it suitable for mixed real-world datasets.

## Highlights

- Zero dependency on ML libraries — built entirely from mathematical first principles
- Handles **mixed feature types**: continuous (Gaussian likelihood) and categorical (Laplace smoothing)
- Log-probability computation for numerical stability (avoids float underflow)
- Full evaluation: accuracy score, confusion matrix, per-class probability output
- Clean OOP design with `fit()`, `predict_probability()`, and `evaluate_on_data()` methods

## Tech Stack

| Component | Technology |
|---|---|
| Implementation | Pure Python |
| Data Handling | pandas |
| Math | Python `math` module |
| Data Structures | `collections.Counter`, `defaultdict` |

## How It Works

```
Training (fit):
  → Compute prior probabilities P(class) for each class
  → For continuous features: compute mean and variance per class (Gaussian)
  → For categorical features: count value frequencies per class

Prediction (predict_probability):
  → For each sample, compute log P(class) + Σ log P(feature | class)
  → Continuous: log-Gaussian likelihood
  → Categorical: Laplace-smoothed probability
  → Convert log-probabilities to normalised probabilities
  → Return predicted class + per-class probabilities
```

## Key Design Decisions

- **Log-probabilities** prevent numerical underflow when multiplying many small probabilities
- **Laplace smoothing** (add-1) handles unseen categorical values gracefully
- **Variance floor** (`1e-9`) prevents division by zero for constant features
- **Mixed feature support** via a `continuous` boolean list passed at initialisation

## Files

| File | Description |
|---|---|
| `naiver_bayes_klassifizierer.py` | Core classifier implementation |
| `NaiveBayes_assignment.ipynb` | Usage examples, dataset loading, evaluation |

## Example Usage

```python
from naiver_bayes_klassifizierer import NaiveBayes

# Define which features are continuous
continuous = [True, False, True, False]  # e.g. [temperature, outlook, humidity, windy]
feature_names = ['Temperature', 'Outlook', 'Humidity', 'Windy']

nb = NaiveBayes(continuous=continuous, feature_names=feature_names)
nb.fit(train_df, target_name='PlayTennis')

accuracy, confusion_matrix, predictions = nb.evaluate_on_data(test_df, 'PlayTennis')
print(f"Accuracy: {accuracy:.2%}")
```

## Course Context

Built as part of the **Data Science Lab** module at Hochschule Heilbronn (Winter Semester 2025/26).
