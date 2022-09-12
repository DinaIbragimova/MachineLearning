import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv(filename):
    data = pd.read_csv(filename)
    return data


def first_task(data):
    first_pclass_survived_male = data[(data['Sex'] == "male") & (data['Survived'] == 1) & (data['Pclass'] == 1)]
    second_pclass_survived_male = data[(data['Sex'] == "male") & (data['Survived'] == 1) & (data['Pclass'] == 2)]
    third_pclass_survived_male = data[(data['Sex'] == "male") & (data['Survived'] == 1) & (data['Pclass'] == 3)]

    first_pclass_survived_female = data[(data['Sex'] == "female") & (data['Survived'] == 1) & (data['Pclass'] == 1)]
    second_pclass_survived_female = data[(data['Sex'] == "female") & (data['Survived'] == 1) & (data['Pclass'] == 2)]
    third_pclass_survived_female = data[(data['Sex'] == "female") & (data['Survived'] == 1) & (data['Pclass'] == 3)]

    female = np.array([len(first_pclass_survived_female), len(second_pclass_survived_female), len(third_pclass_survived_female)])
    male = np.array([len(first_pclass_survived_male), len(second_pclass_survived_male), len(third_pclass_survived_male)])
    width = 0.3
    x = np.arange(3)
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, female, width, label='female')
    ax.bar(x + width / 2, male, width, label='male')
    ax.set_xticks(x)
    ax.set_xticklabels([1, 2, 3])
    ax.legend()

def second_task(data):
    pclasses = [len(data[data['Pclass'] == 1]), len(data[data['Pclass'] == 2]), len(data[data['Pclass'] == 3])]
    fig1, ax1 = plt.subplots()
    ax1.pie(pclasses, labels=[1, 2, 3], autopct='%1.1f%%')
    plt.show()

def third_task(data):
    data['Age'].hist(color="green", label="passengers")
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    dataset = read_csv("train_titanic.csv")
    first_task(dataset)
    second_task(dataset)
    third_task(dataset)

