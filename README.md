# Interpretability and Fairness in Machine Learning

This repo aims to present different techniques for approaching model interpretation and fairness in Machine Learning black-box models.

An explanation of model interpretability techniques can be found in this post: https://medium.com/@fabricio.pretto.c/uncovering-the-magic-interpreting-machine-learning-black-box-models-3154fb8ed01a)

**Techniques covered:**
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

