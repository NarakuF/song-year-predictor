import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import feature_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.decomposition import PCA
from sklearn import neural_network
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor


if __name__ == '__main__':
    data_fn = './MSdata.mat'
    data = loadmat(data_fn)
    X = data['trainx']
    y = data['trainy']

    # show features correlation heatmap
    '''
    plt.figure(figsize=[80, 40])
    sns.heatmap(df.corr(), annot=True)
    plt.show()
    '''

    # show features distribution
    # plt.scatter(X, y)
    # plt.show()

    # remove the less correlated with y: index = 15, 17, 21
    # plt.scatter(X[:20000, 17], X[:20000, 22])
    # plt.show()
    # X = np.delete(X, [15, 17, 21], axis=1)

    '''
    for i in range(X.shape[1]):
        if i % 10 == 0:
            print(i)
        plt.scatter(X[:100000, i], y[:100000])
        plt.xlabel(f'Feature index {i}')
        plt.show()
    plt.close()
    '''

    #vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    #print(vif)
    vif = [0, 3, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    # X = np.delete(X, vif, axis=1)

    # baseline
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # reg = linear_model.LinearRegression()

    # build a ml pipeline
    reg = make_pipeline(# PolynomialFeatures(interaction_only=True),
                        # PCA(n_components='mle'),
                        # SelectPercentile(f_regression, percentile=20),
                        # preprocessing.MinMaxScaler(),
                        # feature_selection.VarianceThreshold(0.001),
                        # SVR())
                        linear_model.LinearRegression())
                        # neural_network.MLPRegressor())
                        # linear_model.SGDRegressor(penalty='l1', l1_ratio=0.8, max_iter=1000, random_state=11, learning_rate='optimal'))
                        # ensemble.GradientBoostingRegressor())

    # train
    # X_train = np.abs(X_train)
    # X_test = np.abs(X_test)
    reg.fit(X_train, y_train)

    # test on the validation set
    y_pred = reg.predict(X_test)

    # round predictions to integer
    # y_pred = np.around(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)

    min_year = np.min(y_test[:, 0])
    max_year = np.max(y_test[:, 0])
    y_mean = np.empty(shape=[max_year - min_year + 1, 2])
    for i in range(min_year, max_year + 1):
        plt.scatter(y_test[y_test[:, 0] == i], y_pred[y_test[:, 0] == i])
        y_mean[i - min_year] = np.array([i, np.mean(y_pred[y_test[:, 0] == i]).item()])
    plt.show()
    plt.scatter(y_mean[:, 0], y_mean[:, 1])
    plt.plot([min_year, max_year], [min_year, max_year])
    plt.show()

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

    # make prediction on the test set
    x_output = data['testx']
    y_output = reg.predict(x_output)
    np.savetxt('output.csv', y_output, delimiter=',')

