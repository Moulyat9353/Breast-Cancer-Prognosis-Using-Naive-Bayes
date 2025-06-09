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
  
# 🔍 Key Results and Insights
**1) Optimal k Selection (Euclidean Distance):**  

🔥 Best performing k = 4  

**Confusion Matrix & Metrics (k = 4):**  

✅ True Positive Rate (Recall): 0.9857  
✅ True Negative Rate: 0.8333  
✅ Precision: 0.9324  
✅ F1 Score: 0.9583  

**2. Performance Across Distance Metrics (Unweighted KNN) with different values of k**  

Minkowski with log-scaled p provides flexible control and performed best in this dataset with **test error of 0.06** at optimal **k-value = 1**  

**3. Weighted KNN Results**  

Euclidean performed best with **test error of 0.10** at optimal **k-value = 6**  

Weighted KNN stabilizes performance across high k values but doesn’t outperform the best unweighted setup.  

# Learning curve  

![Learning Curve](https://github.com/user-attachments/assets/dbb45629-38ee-418e-8da5-76e20930a980)

# 🛠️Technologies Used:  
- Python
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- imblearn for **SMOTE**
