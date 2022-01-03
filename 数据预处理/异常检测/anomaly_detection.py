import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from pyod.models.iforest import IForest # 异常检测模型
from sklearn.preprocessing import StandardScaler
# 模型保存
from joblib import dump, load
import seaborn as sns
import logging

import warnings
warnings.filterwarnings('ignore') 

# def anomaly_detection_(X, outliers_fraction=0.005, Standard=True):
#     '''
#     输入RSP，返回除去异常值的RSP
#     @param: X RSP数据块
#     @param: outliers_fraction 异常值比例
#     '''
#     try:

#         # 这里帮用户标准化了
#         if(Standard):
#             fea = X.columns
#             X[fea] = StandardScaler().fit_transform(X[fea])
#         # 训练异常检测器
#         IF = IForest(
#             contamination=outliers_fraction,
#             random_state=0
#         )
#         IF.fit(X)
#         y_pred = IF.predict(X)
#         X_AD = X.iloc[y_pred != 1, :]# 剔除异常值的好数据
#         return (X_AD, y_pred)
#     except:
#         logging.warning("error")
#     return None
def anomaly_detection_(Block, outliers_fraction=0.005, Standard=True):
    '''
    输入RSP，返回除去异常值的RSP
    @param: X RSP数据块
    @param: outliers_fraction 异常值比例
    '''
    try:
        # 分割df
        X = Block.drop('label', axis=1)
        y = Block['label']
        # 这里帮用户标准化了
        if(Standard):
            fea = X.columns
            ss = StandardScaler()
            X[fea] = ss.fit_transform(X[fea])
        # 训练异常检测器
        IF = IForest(
            contamination=outliers_fraction,
            random_state=0
        )
        IF.fit(X)
        y_pred = IF.predict(X)
        X_AD = X.iloc[y_pred != 1, :]# 剔除异常值的好数据
        y_AD = y.iloc[y_pred != 1]
        X_AD['label'] = y_AD
        return (X_AD, y_pred)
    except:
        logging.warning("异常处理出错")
    return None

def plot_pca(num, data, label):
    '''
    PCA降维可视化函数，可降至3维、2维
    @parma: num 维度
    @parma: data 样本数据（标准化之后?）
    @parma: label 样本标签（正常值，异常值）
    '''
    pca=PCA(n_components=num)
    X_pca=pca.fit_transform(data)
    # print(pca.components_)
    # 分割数据
    X_failure=np.array([x for i,x in enumerate(X_pca) if label[i]==1])
    X_healthy=np.array([x for i,x in enumerate(X_pca) if label[i]==0])
    
    if num==3:
        fig = plt.figure(figsize=[10,15])
        ax = Axes3D(fig)   
        #ax.legend(loc='best')
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        ax.scatter(X_failure[:,0], X_failure[:,1], X_failure[:,2])
        ax.scatter(X_healthy[:,0], X_healthy[:,1], X_healthy[:,2])
    elif num==2:
        plt.figure(figsize=[10,10])
        plt.scatter(X_failure[:,0],X_failure[:,1], label='anomaly')
        plt.scatter(X_healthy[:,0],X_healthy[:,1], label='norm')
        plt.legend()
    else:
        print('i do not want to work.....')
