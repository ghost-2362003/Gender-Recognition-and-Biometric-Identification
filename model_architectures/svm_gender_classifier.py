from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

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
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

pca = PCA(n_components=0.99)
pca_train = pca.fit_transform(X_train)
pca_train = pd.DataFrame(pca_train)
pca_test = pca.transform(X_test)

svm_classifier = SVC(kernel='poly')
param_grid = {
    'gamma': [0.001, 0.01, 0.1, 1, 5, 10],
    'C': [0.001, 0.01, 0.1, 1, 5, 10]
}
grid_search = GridSearchCV(svm_classifier, param_grid=param_grid, cv=10,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(pca_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)