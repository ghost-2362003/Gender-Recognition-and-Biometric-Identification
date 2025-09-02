from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd

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

# creating the SVM classifier
svm_classifier = SVC(kernel='sigmoid',
                     C=1.0, verbose=True)
svm_classifier.fit(pca_train, y_train)

# making predictions
svm_preds = svm_classifier.predict(pca_test)
svm_preds