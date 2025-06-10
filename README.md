# 🧠Problem Statement:  

The objective of this project is to **predict the recurrence of breast cancer** in patients using features extracted from digital images of Fine Needle Aspirates (FNA) of breast masses. Each image is processed to extract 30 real-valued attributes that quantify various characteristics of cell nuclei, such as shape, texture, and structure. These features, along with two additional clinical attributes, are used to train a **Naive Bayes classifier** to determine whether a patient's cancer is likely to recur within two years.

Due to the class imbalance in the dataset—more non-recurrent cases than recurrent— **Synthetic Minority Oversampling Technique (SMOTE)** is also applied to assess whether balancing the training data improves model performance.

# 📂 Dataset:
**Source:** WPBC Data Set from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Attributes:** 30 real-valued features extracted from digital FNA images + 2 additional clinical features.  

**Total Instances:** 198 - 151 non recurrent cases and 47 recurrent cases.

**Labels:** R -> Recurrent and N -> Non Recurrent.  

# ⚙️ Task Overview:
**1) Data Preprocessing:**  
- Selected the first 130 non-recurrent and first 37 recurrent cases as training data.

- Added record #197 to the training set.

- Replaced missing lymph node counts with the median value of the training set.  

**2) Modelling:**  
- Implemented a Gaussian Naive Bayes classifier.

- Evaluated performance using: Confusion matrix, Precision, Recall, F1-score, ROC and AUC.  

**3) SMOTE for imbalanced classes:**  
- Applied SMOTE (Synthetic Minority Oversampling Technique): Downsampled the majority class to 90 samples and upsampled the minority class to 90 samples.

- Re-evaluated the model on both the train and original test sets.
  
# 🔍 Results:
| Metric    | Before SMOTE (Train Set) | After SMOTE (Train Set) |
| --------- | ----------------------- | ---------------------- |
| Precision | `0.34`                 | `0.7179`                |
| Recall    | `0.49`                 | `0.6222`                |
| F1 Score  | `0.40`                 | `0.6666`                |
| AUC       | `0.6845`                 | `0.7675`                |  

| Metric    | Before SMOTE (Test Set) | After SMOTE (Test Set) |
| --------- | ----------------------- | ---------------------- |
| Precision | `0.42`                 | `0.0`                |
| Recall    | `0.50`                 | `0.0`                |
| F1 Score  | `0.45`                 | `0.0`                |
| AUC       | `0.5862`                 | `0.6552`                |

SMOTE did not help in this scenario as it made the model better at recognizing the synthetic oversampled minority class but failing on actual unseen minority-class samples. Alternative solutions like cost-sensitive learning, or hybrid approaches might be better for handling class imbalance while maintaining generalization.

# 🛠️Technologies Used:  
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- imblearn for **SMOTE**
