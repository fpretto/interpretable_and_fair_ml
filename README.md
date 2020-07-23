# Interpretability and Fairness in Machine Learning

This repo aims to present different techniques for approaching model interpretation and fairness in Machine Learning black-box models.

Most of the *model interpretation* techniques were taken from this highly recommendable book from Christoph Molnar:
**Interpretable Machine Learning -** ***A Guide for Making Black Box Models Explainable***. (https://christophm.github.io/interpretable-ml-book/)

For model *fairness assessment*, a practical guide can be found in this blog post:
**FairML: Auditing Black-Box Predictive Models** (https://blog.fastforwardlabs.com/2017/03/09/fairml-auditing-black-box-predictive-models.html)

**Techniques used:**
- **Model Interpretation**
  - Global Importance
    1. Feature Importance (evaluated by the XGBoost model and by SHAP)
    2. Summary Plot (SHAP)
    3. Permutation Importance (ELI5)
    4. Partial Dependence Plot (evaluated by PDPBox and by SHAP)
    5. Global Surrogate Model (Decision Tree and Logistic Regression)
  - Local Importance
    1. Local Interpretable Model-agnostic Explanations (LIME)
    2. SHapley Additive exPlanations (SHAP)
        - Force Plot
        - Decision Plot

- **Model Fairness**
  - FairML

