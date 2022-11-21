from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

iris_dataset = datasets.load_iris()

iris = pd.DataFrame(
    data=np.c_[iris_dataset['data'], iris_dataset['target']],
    columns=iris_dataset['feature_names'] + ['target']
)

iris.head()

iris.isnull().sum()

def normalize_column(initial_data):
    data = np.copy(initial_data)
    min = data.min()
    max = data.max()
    for i in range(len(data)):
        data[i] = (data[i] - min) / (max - min)
    return data


def normalize_dataset(data):
    normal = np.stack([
        normalize_column(data.iloc[:, 0]),
        normalize_column(data.iloc[:, 1]),
        normalize_column(data.iloc[:, 2]),
        normalize_column(data.iloc[:, 3]),
        data.iloc[:, 4]
    ])
    return pd.DataFrame(normal.transpose(), columns=iris.columns)


normal_iris = normalize_dataset(iris)
normal_iris.tail()

di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}  # dictionary

before = sns.pairplot(iris.replace({'target': di}), hue='target')
before.fig.suptitle('Pair Plot of the dataset Before normalization', y=1.08)

after = sns.pairplot(normal_iris.replace({'target': di}), hue='target')
after.fig.suptitle('Pair Plot of the dataset After normalization', y=1.08)


def train_test_split(data):
    N = len(data)
    train_count = int(round(len(data) * 0.75))
    test_count = N - train_count;

    random_iris = iris.sample(frac=1).reset_index(drop=True)
    target = random_iris['target'];
    random_iris.drop(['target'], axis=1)

    train = random_iris.head(train_count)
    train_target = target[:train_count]
    test = random_iris.tail(test_count)
    test_target = target[train_count:]

    return train, test, train_target, test_target


x = iris.iloc[:, :-1]  # все параметры
y = iris.iloc[:, -1]  # признак который хотим определить

x_train, x_test, y_train, y_test = train_test_split(iris)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_n = normal_iris.iloc[:, :-1]  # все параметры
y_n = normal_iris.iloc[:, -1]  # признак который хотим определить

x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(normal_iris)
x_train_n = np.asarray(x_train_n)
y_train_n = np.asarray(y_train_n)

x_test_n = np.asarray(x_test_n)
y_test_n = np.asarray(y_test_n)


def get_distances(x_train, x_test_point):
    distances = []
    for point in x_train:
        current_distance = 0
        for i in range(len(point)):
            current_distance += (point[i] - x_test_point[i]) ** 2
        current_distance = np.sqrt(current_distance)

        distances.append(current_distance)
    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


def nearest_neighbors(distance_point, K):
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    return df_nearest[:K]


def voting(df_nearest, y_train):
    counter_vote = Counter(y_train[df_nearest.index])

    y_pred = counter_vote.most_common()[0][0]

    return y_pred


result = []
K = int(round(np.sqrt(x_train.shape[0])))
for x_test_point in x_test:
    distances = get_distances(x_train, x_test_point)
    df_nearest_point = nearest_neighbors(distances, K)
    possible_y = voting(df_nearest_point, y_train)
    result.append(possible_y)

print(result)
print(y_test)

result = []
K = int(round(np.sqrt(x_train_n.shape[0])))
for x_test_point in x_test_n:
    distances = get_distances(x_train_n, x_test_point)
    df_nearest_point = nearest_neighbors(distances, K)
    possible_y = voting(df_nearest_point, y_train_n)
    result.append(possible_y)

print(result)
print(y_test_n)

point = [5.1, 3.5, 1.4, 0.2]
distance_point = get_distances(x_train, point)
df_nearest_point = nearest_neighbors(distance_point, K)
possible_y = voting(df_nearest_point, y_train)

print(possible_y)
