
def clf_metrics(y_pred_train, y_proba_train, y_train, y_pred_test, y_proba_test, y_test):
    """ Generates dataframe with kpis for classification models

        Parameters:
        y_pred_train: model binary train prediction
        y_proba_train: model train probability prediction
        y_train : model true train target values
        y_pred_test: model binary test prediction
        y_proba_train: model  probability test prediction
        y_test : model true test target values

        Returns:
        dataframe  with Accuracy, Precision, Recall, F1, AUC for train and test predictions
       """

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
    import pandas as pd
    import matplotlib.pyplot as plt

    y_pred_train = pd.Series(y_pred_train).reset_index(drop=True).copy()
    y_proba_train = pd.Series(y_proba_train).reset_index(drop=True).copy()
    y_train = pd.Series(y_train).reset_index(drop=True).copy()

    y_pred_test = pd.Series(y_pred_test).reset_index(drop=True).copy()
    y_proba_test = pd.Series(y_proba_test).reset_index(drop=True).copy()
    y_test = pd.Series(y_test).reset_index(drop=True).copy()

    fpr_train, tpr_train, threshold_train = roc_curve(y_train, y_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, threshold_test = roc_curve(y_test, y_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Results

    train_accuracy = round(accuracy_score(y_train, y_pred_train), 2)
    train_precision = round(precision_score(y_train, y_pred_train), 2)
    train_recall = round(recall_score(y_train, y_pred_train), 2)
    train_f1 = round(f1_score(y_train, y_pred_train), 2)

    test_accuracy = round(accuracy_score(y_test, y_pred_test), 2)
    test_precision = round(precision_score(y_test, y_pred_test), 2)
    test_recall = round(recall_score(y_test, y_pred_test), 2)
    test_f1 = round(f1_score(y_test, y_pred_test), 2)

    train_results = pd.concat([pd.Series(train_accuracy), pd.Series(train_precision), pd.Series(train_recall),
                               pd.Series(train_f1), pd.Series(roc_auc_train)], axis=1)

    test_results = pd.concat([pd.Series(test_accuracy), pd.Series(test_precision), pd.Series(test_recall),
                              pd.Series(test_f1), pd.Series(roc_auc_test)], axis=1)

    results = train_results.append(test_results)
    results.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    results.index = ['Train', 'Test']

    # Curva ROC
    # Train
    plt.title('TRAIN: ROC Curve')
    plt.plot(fpr_train, tpr_train, 'b', label = 'AUC = %0.2f' % roc_auc_train)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # Test

    plt.title('TEST: ROC Curve')
    plt.plot(fpr_test, tpr_test, 'b', label = 'AUC = %0.2f' % roc_auc_test)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return results