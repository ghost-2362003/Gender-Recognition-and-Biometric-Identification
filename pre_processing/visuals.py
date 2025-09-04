from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
                       cmap='Purples', xticklabels=['M', 'F'], 
                       yticklabels=['M', 'F'])
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.show() 
    
def generate_reports(y_pred, y_true, proba):
    report = classification_report(y_pred, y_true)
    accuracy = accuracy_score(y_true, y_pred)
    curves = roc_auc_score(y_true, proba)
    
    return report, accuracy, curves