# thyroid-cancer-risk-ml
Thyroid Cancer Risk Analysis using Machine Learning: Classification, Clustering , Feature Correlations

This project develops a machine learning pipeline to assess thyroid cancer risk, identify critical predictive features, and support clinical decision-making through data-driven insights.

📊 Dataset

212,691 patient records

17 clinical & demographic features

Includes thyroid function indicators (TSH, T3, T4, nodules, etc.)

Class imbalance present → addressed via undersampling

🧹 Data Pre-Processing
Step	Technique	Purpose
Cleaning	Null handling & filtering	Reliable input for models
Normalization	StandardScaler	Improve clustering geometry
Class Imbalance Fix	Subsampling	Prevent majority class dominance
Model Evaluation	Stratified 4-Fold Cross-Validation	Fair results across splits

📌 Models evaluated 100 times → Final scores = Median Accuracy + Median AUC

🔥 Correlation Analysis

Heatmap used to inspect feature relationships

Highlighted strong relationships among hormone indicators

Guided feature importance interpretation & model focus

🧠 Machine Learning Modeling
Algorithms

Logistic Regression

Random Forest Classifier

Feature Importance

Performed using two independent methods:

Model	Method	What it tells us
Logistic Regression	Coefficients	Feature direction & significance
Random Forest	Impurity-based importance	Non-linear influence

➡ Extracted top-ranked features
➡ Formed triad feature subsets
➡ Evaluated using CV ranking by median accuracy → median AUC

🔍 Clustering Insights (K-Means)

K = 3 clusters selected

PCA applied for visualization

Distinct patient patterns observed → possible risk grouping

🎛 GUI Deployment

A prediction interface built using Tkinter:

Inputs:

Age, gender, hormones, risk indicators

Output:

Low / Medium / High cancer risk classification

Demonstrates practical usability & accessibility.

⚙️ Tech Stack
Category	Tools
Data Processing	Python, Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML Modeling	Scikit-learn, Weka
GUI	Tkinter
Notebooks	JupyterLab
📈 Results & Conclusions

✔ Balanced predictive performance
✔ Random Forest triads outperform LR
✔ Robust due to repeated stratified CV
✔ Clustered sub-populations show potential in medical segmentation
✔ Deployable GUI shows real healthcare applicability

📁 Repository Structure
thyroid-cancer-risk-ml/
├── notebooks/
├── visuals/
│   ├── correlation_heatmap.png
│   ├── feature_importance_rf.png
│   ├── feature_importance_lr.png
│   ├── clusters_pca.png 
│   └── gui_app.png
├── app/
│   └── thyroid_gui.py
└── README.md

🧩 Future Improvements

Hyperparameter tuning

Synthetic oversampling (e.g., SMOTE)

Additional clinical biomarkers

Model explainability dashboards

👩‍⚕️ Author

Vasileia Damaskou Sutton
Junior Data Analyst | Healthcare Analytics
Python · SQL · Power BI · Tableau · Machine Learning · Weka

✳️ This project is part of my MSc Business Information Systems thesis work.✔ Deployable interface → demonstrates practical utility
