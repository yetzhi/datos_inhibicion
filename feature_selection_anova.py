from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
import pandas as pd

# feature selection
def select_features(X_train, y_train, X_test):
	# configure to select all features
	fs = SelectKBest(score_func=f_classif, k='all')
	# learn relationship from training data
	fs.fit(X_train, y_train)
	# transform train input data
	X_train_fs = fs.transform(X_train)
	# transform test input data
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs
 
# load the dataset
df = pd.read_csv("g2-r7.csv")
X = df.drop(['class'], axis=1)
y = df['class']
X = X.values
y = y.values
fs_actualizado = []
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	if (fs.scores_[i] > 1.5):
		print('Feature %d: %f' % (i+1, fs.scores_[i]))
		fs_actualizado.append(fs.scores_[i])
# plot the scores
pyplot.bar([i for i in range(len(fs_actualizado))], fs_actualizado)
pyplot.show()
