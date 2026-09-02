# 🌳 Esercizio: Decision Tree, Gestione dello Sbilanciamento e Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gTngdEr8IBNHr0mEMh2bGq7H0qaA6mVq?usp=sharing)

## 📝 Descrizione
In questo progetto, contenuto nel notebook `01_decision_tree_2.ipynb`, esploro l'addestramento e l'ottimizzazione di un modello **Decision Tree** (Albero Decisionale) utilizzando il celebre dataset Iris. 

L'obiettivo principale non è solo la semplice classificazione, ma affrontare problemi pratici del Machine Learning, come la **gestione delle classi sbilanciate** e la **prevenzione dell'overfitting** attraverso la ricerca dei migliori iperparametri.

## 🎯 Competenze dimostrate e Flusso di Lavoro
Il notebook è strutturato nei seguenti passaggi chiave:
1. **Valutazione del Modello Base**: Addestramento con *Stratified 5-Fold Cross-Validation* e analisi delle performance tramite Matrice di Confusione e Classification Report.
2. **Analisi ROC Multiclasse**: Binarizzazione del target (approccio *One-vs-Rest*) per tracciare le curve ROC e calcolare l'AUC per ciascuna classe.
3. **Gestione Classi Sbilanciate (Oversampling)**: "Inflazione" manuale (replica 10x) della classe *Virginica* nel training set per testare il comportamento dell'algoritmo in presenza di una classe fortemente predominante.
4. **Cost-Sensitive Learning**: Utilizzo del parametro `class_weight` per penalizzare gli errori sulla classe *Virginica* senza alterare fisicamente il dataset, dimostrando l'equivalenza matematica (100% di match) con il metodo dell'inflazione manuale.
5. **Hyperparameter Tuning**: Utilizzo di `GridSearchCV` per esplorare quasi 40.000 combinazioni di iperparametri in parallelo, al fine di ridurre l'overfitting e trovare l'architettura ottimale dell'albero.
6. **Visualizzazione Avanzata**: Plot affiancato della Matrice di Confusione finale e della struttura grafica dell'albero decisionale ottimizzato.

## 🛠️ Tecnologie Utilizzate
* **Python**
* **Scikit-Learn** (DecisionTreeClassifier, GridSearchCV, StratifiedKFold, Metriche di Valutazione)
* **Matplotlib** (Visualizzazione grafici, Curve ROC, Plot dell'albero)
* **NumPy** (Manipolazione array e campionamento)

## 🚀 Come visualizzare ed eseguire l'esercizio
Il modo più rapido per esplorare il codice è cliccare sul badge **"Open in Colab"** in cima a questa pagina. 
L'ambiente si aprirà direttamente nel tuo browser, preconfigurato e pronto all'uso. Il file si aprirà in sola lettura: potrai eseguire tutte le celle per vedere i risultati, ma se desideri fare esperimenti o modificare il codice, ti basterà cliccare su `File > Salva una copia in Drive`.