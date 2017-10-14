import pandas as pd;
import seaborn as sns;
from sklearn.preprocessing import StandardScaler;
from sklearn.cross_validation import train_test_split;
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

dataset = pd.read_csv("Data/train.csv");

# shape
shape = dataset.shape
print(shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('Survived').size())

# box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# histograms
names = dataset.columns.values
#for x in range(0,shape[1]):
for name in names:
    try:
        # data = dataset.ix[:, x]
        data = dataset[name]
        # data.hist()
        plt.hist(data, bins='auto')

        #plt.xlabel('time (s)')
        #plt.ylabel('voltage (mV)')
        plt.title('Histogram of ' + name)
        plt.grid(True)
        plt.savefig('Image/' + name + 'Hist.png')
        plt.show()
    except:
        print "Error with data"

# scatter plot matrix
scatter_matrix(dataset)
#plt.show()
plt.savefig('Image/dataset.png')

# test_well = dataset[dataset['Well Name'] == 'SHANKLE'];
# data = dataset[dataset['Well Name'] != 'SHANKLE'];
features = ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare','Embarked']
feature_vectors = dataset[features];
# facies_labels = dataset['Facies'];

# sns.pairplot(feature_vectors[['Age', 'ILD_log10', 'DeltaPHI', 'PHIND','PE']]);
# sns.pairplot(feature_vectors);

# scaler = StandardScaler().fit(feature_vectors);
# scaled_features = scaler.transform(feature_vectors);
# X_train, X_cv, y_train, y_cv = \
# train_test_split(scaled_features, facies_labels,
# test_size=0.05, random_state=42)
#
# clf = svm.SVC(C=10, gamma=1);
# clf.fit(X_train, y_train);
#
# y_test = test_well['Facies'];
# well_features = test_well.drop(['Facies',
# 'Formation',
# 'Well Name',
# 'Depth'],
# axis=1);
#
# X_test = scaler.transform(well_features);
# y_pred = clf.predict(X_test);
# test_well['Prediction'] = y_pred;
#
# target_names = ['SS', 'CSiS', 'FSiS', 'SiSh',
# 'MS', 'WS', 'D','PS', 'BS']
# print(classification_report(y_test, y_pred,
# target_names=target_names))
