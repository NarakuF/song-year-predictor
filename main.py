import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.decomposition import PCA
from sklearn import neural_network


if __name__ == '__main__':
    data_fn = './MSdata.mat'
    data = loadmat(data_fn)
    X = data['trainx']
    y = data['trainy']

    df = pd.DataFrame(np.concatenate((X[:, :23], y), axis=1))
    df_y = pd.DataFrame(y)

    # show features correlation heatmap
    # plt.figure(figsize=[80, 40])
    # sns.heatmap(df.corr(), annot=True)
    # plt.show()

    # remove the less correlated with y: index = 15, 17, 21
    # X = np.delete(X, [15, 17, 21], axis=1)

    # show features distribution
    # plt.plot(X, y)
    # plt.show()

    # baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # reg = linear_model.LinearRegression()

    # build a ml pipeline
    reg = make_pipeline(# PolynomialFeatures(),
                        # PCA(n_components='mle'),
                        # SelectPercentile(f_regression, percentile=20),
                        StandardScaler(),
                        neural_network.MLPRegressor())
                        # linear_model.SGDRegressor(penalty='l1', l1_ratio=0.8, max_iter=1000, random_state=11, learning_rate='optimal'))
                        # linear_model.LassoCV(cv=10))

    # train
    reg.fit(X_train, y_train)

    # test on the validation set
    y_pred = reg.predict(X_test)

    # round predictions to integer
    # y_pred = np.around(y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # comparisons
    # LinearRegression()                    6.760205411939791
    # LinearRegression() around             6.755269939510259
    # LinearRegression() StandardScaler     6.760205411939838
    # LassoCV(cv=10)                        6.819801902438152
    # LassoCV(cv=10) StandardScaler         6.760077770085714
    # LassoCV(cv=10) StandardScaler around  6.755614979028067
    # RidgeCV(cv=10)                        6.760205901782357
    # RidgeCV(cv=10) StandardScaler         6.760224341915719
    # RidgeCV(cv=10) StandardScaler around  6.755334634419849
    # ElasticNetCV(cv=10)                   6.852804856414208
    # ElasticNetCV(cv=10) StandardScaler    6.761995903730985
    # DecisionTreeRegressor(max_depth=5)    7.168562562134506
    # SGDRegressor(penalty='l1', l1_ratio=0.8, max_iter=1000, random_state=11, learning_rate='optimal')6.762203201213515
    # MLPRegressor()                        6.585188772812935
    print(mae)

    # make prediction on the test set
    x_output = data['testx']
    y_output = reg.predict(x_output)
    np.savetxt('output.csv', y_output, delimiter=',')

