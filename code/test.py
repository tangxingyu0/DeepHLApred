from evaluate_metrics import sn_sp_acc_mcc
from model import getmodel
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def predict_model(x_test, y_test, weights):
    model = getmodel()
    model.load_weights(weights)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    AUC = auc(fpr, tpr)
    print(AUC)
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    sn_sp_acc_mcc(y_test, y_pred)

    plt.figure()
    plt.title('HLA-E*01:01')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.plot(0, 0, color='b', label=r'Mean ROC (AUROC=%0.4f)' % auc_sum, lw=1, alpha=0.8)
    # plt.plot(fpr5, tpr5, label='ROC5(AUROC=%0.4f)' % (AUC5), alpha=0.8)
    # plt.plot(fpr4, tpr4, label='ROC4(AUROC=%0.4f)' % (AUC4), alpha=0.8)
    # plt.plot(fpr3, tpr3, label='ROC3(AUROC=%0.4f)' % (AUC3), alpha=0.8)
    # plt.plot(fpr2, tpr2, label='ROC2(AUROC=%0.4f)' % (AUC2), alpha=0.8)
    plt.plot(fpr, tpr, label='ROC1(AUROC=%0.4f)' % (AUC), alpha=0.8)
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.legend(loc='lower right')
    # plt.savefig('五折交叉验证E0101', dpi=300)
    plt.show()


