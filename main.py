from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import math


def seq_to_vector(seq):
    vector = np.zeros(780)
    for i in range(0, len(seq)):
        vector[i] = int(seq[i])
    k = 20
    for i in range(0, 19):
        for j in range(i + 1, 20):
            vector[k + int(seq[i]) * 2 + int(seq[j])] = 1
            k = k + 4
    return vector


def index_to_matrix(column):
    matrix = np.zeros([4987, 780])
    column = column.to_numpy()
    for l in range(0, 4987):
        for i in range(0, len(column[l])):
            matrix[l, i] = int((column[l])[i])
        k = 20
        for i in range(0, 19):
            for j in range(i + 1, 20):
                matrix[l, k + int((column[l])[i]) * 2 + int((column[l])[j])] = 1
                k = k + 4
    return matrix


def model_classifier(df, pred, resp):
    # define predictor and response variables
    X = df.loc[pred]
    y = df.loc[resp]

    # define cross-validation method to use
    cv = LeaveOneOut()

    # build multiple linear regression model
    model = LinearRegression()

    # use LOOCV to evaluate model
    scores = cross_val_score(model, X.T, y.T, scoring='neg_mean_absolute_error',
                             cv=cv, n_jobs=-1)

    # view mean absolute error
    return np.mean(np.absolute(scores))


if __name__ == '__main__':
    data = pd.read_csv("df.csv", dtype='str')
    data = data.set_index('Unnamed: 0')
    data = data.rename_axis(index=None, columns=None)
    data = data.astype('Int64')
    sum_of_obs = data.sum(axis=0)
    indices: list[str] = []
    for i in range(0, 20):
        indices.append(str(i))
    for i in range(0, 19):
        for j in range(i + 1, 20):
            indices.append(str(i) + "_" + str(j) + "_" + "00")
            indices.append(str(i) + "_" + str(j) + "_" + "01")
            indices.append(str(i) + "_" + str(j) + "_" + "10")
            indices.append(str(i) + "_" + str(j) + "_" + "11")
    df = pd.DataFrame(0, columns=data.columns, index=indices)

    matrix = index_to_matrix(data.index)


    def count_freq_for_p(column):
        seq_weights = data[column.name] / sum_of_obs.loc[column.name]
        l = np.matmul(np.transpose(matrix), seq_weights)
        column = column + l
        return column


    df = df.apply(count_freq_for_p, axis=0)
    df = df.applymap(math.log1p)
    dfcol = data.columns
    canc = np.zeros([1, 30])
    for i in range(0, 30):
        if (dfcol[i])[4] == 't':
            canc[0, i] = 0
        if (dfcol[i])[4] == 'c':
            canc[0, i] = 1
    newrow = pd.DataFrame(canc, columns=data.columns, index=["Cancer"])
    df = pd.concat([newrow, df])
    print(model_classifier(df, ['0_2_00', '18_19_11'], 'Cancer'))
