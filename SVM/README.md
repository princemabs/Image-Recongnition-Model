# **Guide d'Inférence - Modèle SVM One-vs-One pour MNIST**

## **Table des matières**
1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Guide d'utilisation](#guide-dutilisation)
4. [Exemples de code](#exemples-de-code)

---

## Vue d'ensemble

Ce modèle implémente une **classification SVM multi-classe utilisant la stratégie Un-Contre-Un (One-vs-One)** pour la reconnaissance de chiffres manuscrits MNIST (0-9).

### Caractéristiques principales:
- **Stratégie**: One-vs-One (45 classificateurs binaires pour 10 classes)
- **Kernel**: Polynômial (degré 2)
- **Vote majoritaire**: Chaque SVM exprime un vote
- **Optimisation**: Quadratic Programming (cvxopt)
- **Scalabilité**: Entraîné sur 1000 samples, testable sur tout ensemble MNIST

---

## Prérequis

### Python 3.8+
```bash
python --version  # Vérifier la version
```

### Bibliothèques requises:
```
numpy
pandas
scikit-learn
cvxopt
matplotlib
```

## Guide d'utilisation

### Option 1: Utiliser le notebook complet

1. Ouvrir `INF372_TP5_Groupe-01.ipynb` dans Jupyter/VS Code
2. Exécuter toutes les cellules pour:
   - Charger les données MNIST
   - Prétraiter l'ensemble de données
   - Entraîner le modèle
   - Évaluer les performances

### Option 2: Script Python standalone

Créer un fichier `inference.py`:

```python
import numpy as np
import pandas as pd
import cvxopt
from sklearn.metrics import accuracy_score, classification_report

# ====== Charger vos données ======
# X_train, y_train: données et labels d'entraînement
# X_test, y_test: données et labels de test

# Normaliser les données (0-255 → 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# ====== Importer le modèle depuis le notebook ======
# Copier les classes SVM et OneVsOneMultiClassSVM du notebook

# ====== Entraîner le modèle ======
model = OneVsOneMultiClassSVM(kernel='polynomial', degree=2, C=1.0)
model.fit(X_train, y_train)

# ====== Faire des prédictions ======
y_pred = model.predict(X_test)

# ====== Évaluer ======
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

---

## Exemples de code

### Exemple 1: Faire une prédiction simple
```python
# Une image de 28x28 pixels aplatie en vecteur (784,)
single_image = X_test[0].reshape(1, -1)  # Shape: (1, 784)
single_image = single_image / 255.0       # Normaliser

# Prédire
prediction = model.predict(single_image)
print(f"Prédiction: {prediction[0]}")
```

### Exemple 2: Prédire plusieurs images
```python
# 10 images test
test_batch = X_test[:10] / 255.0

# Prédictions
predictions = model.predict(test_batch)

# Comparer avec les vraies valeurs
for i, pred in enumerate(predictions):
    print(f"Image {i}: Prédicted={pred}, True={y_test[i]}")
```