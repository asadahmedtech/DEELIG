# import pandas as pd 
# affinities = '0_7-predictions.csv'
# df = pd.read_csv(affinities)

# print(df.head())

# print('RMSE: ', ((df.real - df.predicted)**2).mean() ** 0.5)
# print('PCC: ', df.corr('pearson'))

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import r2_score, mean_squared_error

# casf = open('casf2013.txt', 'r')
# casf = casf.readlines()
# casf = [i[:4].upper() for i in casf]
# print(casf[0])

# casfdf = pd.DataFrame(casf, columns=['set'])
# print(casfdf.head())

# casfresdf = df[df.pdbid.map(lambda x: np.isin(x[:4], casf).all())]
# print(casfresdf)

# # for i in df.set:
# #     print(i)
# #     if(i[:4] in casf):
# #         print(i)
def r2_rmse( g ):
    r2 = scipy.stats.pearsonr( g['real'], g['predicted'] )[0]
    # r2 = df.corr('pearson')
    rmse = np.sqrt( mean_squared_error( g['real'], g['predicted'] ) )
    mse = np.mean(np.abs( g['real']- g['predicted'] ))
    # sd = np.sum()
    return pd.Series( dict(  PCC = r2, mse = mse,rmse = rmse, count=len(g)) )

# print("Our Model")
# print(df.groupby( 'set' ).apply( r2_rmse ).reset_index().to_csv('a.csv'))
# print()
# print("PDBBIND Coreset v2013")
# print(casfresdf.groupby( 'set' ).apply( r2_rmse ).reset_index().to_csv('b.csv'))

# print("PDBBIND Coreset v2013")
# print(r2_rmse(casfresdf).to_csv('b1.csv'))

# casf = open('casf2016.txt', 'r')
# casf = casf.readlines()
# casf = [i[:4].upper() for i in casf]
# print(casf[0])

# casfdf = pd.DataFrame(casf, columns=['set'])
# print(casfdf.head())

# casfresdf = df[df.pdbid.map(lambda x: np.isin(x[:4], casf).all())]
# print(casfresdf)

# print()
# print("PDBBIND Coreset v2016")
# print(casfresdf.groupby( 'set' ).apply( r2_rmse ).reset_index().to_csv('c.csv'))

# print("PDBBIND Coreset v2016")
# print(r2_rmse(casfresdf).to_csv('c1.csv'))

import os
files = os.listdir()

resdf = pd.DataFrame()

for f in files:
    print(f[-4:])
    if(f[-4:] != '.csv'):
        continue
    df = pd.read_csv(f)
    df = df.groupby( 'set' ).apply( r2_rmse ).reset_index()
    filename = f.split('-')[0].split('_')[1]
    print(filename)
    df["file"] = [filename for i in range(3)]
    resdf = resdf.append(df, ignore_index=True)

print(resdf.to_csv("out.csv"))
