from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
from joblib import load, dump
from pre_processing.visuals import generate_heatmap, generate_reports

# creating the dataframes
train_data = pd.read_csv('dataset/train_features.csv')
test_data = pd.read_csv('dataset/test_features.csv')

test_labels = test_data['Label']
test_data.drop(columns=['Label'], inplace=True)

# necessary pre-processing and feature extraction
train_data['Label'] = train_data['Label'].apply(lambda x: 'F' if x == '[0]' else 'M')
X_train, X_test, y_train, y_test = train_test_split(train_data.drop(columns=['Label']),
                                                    train_data['Label'], random_state=42,
                                                    shuffle=True, stratify=train_data['Label'])

pca = PCA(n_components=0.99)
pca_train = pca.fit_transform(X_train)
pca_train = pd.DataFrame(pca_train)
pca_test = pca.transform(X_test)

'''
svm_classifier = SVC(C=9, gamma=0.001, kernel='rbf')
svm_classifier.fit(pca_train, y_train)
dump(svm_classifier, 'model_files/svm_classifier.jobllib')
'''

svm_classifier = load('model_files/svm_classifier.jobllib')
y_pred = svm_classifier.predict(pca_test)

generate_heatmap(y_test, y_pred)
generate_reports(y_pred, y_test, svm_classifier.decision_function(pca_test))