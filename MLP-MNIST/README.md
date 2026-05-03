#  MNIST Perceptron : Guide d'Inférence

Ce projet utilise un Perceptron Multicouche pour reconnaître des chiffres manuscrits. Vous pouvez tester le modèle avec vos propres images en suivant ces étapes.

##  Pré-requis
Assurez-vous d'avoir les bibliothèques suivantes installées :
- `tensorflow`
- `pandas`
- `matplotlib`
- `seaborn`
- `Pillow` (pour la gestion des images externes)

##  Comment tester vos propres chiffres ?

1. **Préparez votre image** :
   - Dessinez un chiffre noir sur une feuille blanche (ou l'inverse).
   - Prenez une photo ou faites une capture d'écran.
   - Enregistrez-la au format `.png` ou `.jpg`.

2. **Placez l'image** :
   - Mettez votre fichier image dans le même dossier que le notebook `.ipynb`.

3. **Exécutez l'inférence** :
   - Allez à la dernière cellule du notebook.
   - Remplacez le nom du fichier par le vôtre :
     ```python
     predire_mon_image('mon_chiffre.png')
     ```
   - Exécutez la cellule.  .

##  Comprendre le résultat
Le script va automatiquement :
1. **Redimensionner** votre image en 28x28 pixels.
2. **Inverser les couleurs** (pour correspondre au format MNIST : fond noir, trait blanc).
3. **Afficher la Heatmap** : Vous verrez en temps réel quels neurones de la couche cachée ont été stimulés par votre écriture.

---
*Projet INF372 - Université de Yaoundé 1*