# -*- coding: utf-8 -*-

import pandas as pd
import os
import time
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import GridSearchCV
from sklearn import metrics

start_time = time.time()
pd.set_option("display.max_rows", None, "display.max_columns", None)

#Define diretório raiz dos arquivos de entrada
os.chdir('D:\GDRIVE\TCC\Datasets')

#Carrega os dados do CSV consolidado para o Pandas DataFrame
print('CARREGANDO ARQUIVO CONSOLIDADO')
df = pd.read_csv('saidaTotal.csv', sep=';', encoding='ISO-8859-1')

#Transforma colunas de respostas do questionário em int64 - Vieram como float do CSV
for i in range(1,26):
    if(i<10):
        df['Q00'+str(i)] = df['Q00'+str(i)].apply(np.int64)
    else:
        df['Q0'+str(i)] = df['Q0'+str(i)].apply(np.int64)

#Verifica se existe algum valor nulo ou duplicado no DataFrame - Não Existe!
## NULOS
print(df.isnull().sum())
## DUPLICADOS
print(df.duplicated().sum())

df.info(verbose=True, null_counts=True)

#Imprime a menor e a maior nota do dataset completo
print(df['MEDIA_FINAL'].min())
print(df['MEDIA_FINAL'].max())

#Plota Histograma Frequência
sns.histplot(df['MEDIA_FINAL'],color='blue')
plt.legend(labels=['MEDIA_FINAL'], ncol=1, loc='upper left');

#Imprime as maiores médias do dataset completo
print(df['MEDIA_FINAL'].value_counts().head())

#Separa as notas em faixas para que sirvam de balanceamento dos dados
df.loc[df['MEDIA_FINAL'] < 390, 'CLUSTER'] = 1
df.loc[(df['MEDIA_FINAL'] >= 390) & (df['MEDIA_FINAL'] < 500), 'CLUSTER'] = 2
df.loc[(df['MEDIA_FINAL'] >= 500) & (df['MEDIA_FINAL'] < 690), 'CLUSTER'] = 3
df.loc[df['MEDIA_FINAL'] >= 690, 'CLUSTER'] = 4

#Imprime a quantidade de dados em cada cluster
print(df['CLUSTER'].value_counts())

#Balanceia o dataframe, pegando apenas 100 mil linhas para cada cluster - Total de 400 mil linhas
dfc1 = df.loc[df['CLUSTER'] == 1].sample(100000, random_state=42)
dfc2 = df.loc[df['CLUSTER'] == 2].sample(100000, random_state=42)
dfc3 = df.loc[df['CLUSTER'] == 3].sample(100000, random_state=42)
dfc4 = df.loc[df['CLUSTER'] == 4].sample(100000, random_state=42)

df_inicial = pd.concat([dfc1, dfc2, dfc3, dfc4], axis=0)

#Verifica se existe algum valor nulo ou duplicado no novo DataFrame - Não Existe!
## NULOS
print(df_inicial.isnull().sum())
## DUPLICADOS
print(df_inicial.duplicated().sum())

#Imprime as maiores médias do dataset balanceado
print(df_inicial['MEDIA_FINAL'].value_counts().head(8))

#Plota Histograma Frequência do dataset balanceado
sns.histplot(df_inicial['MEDIA_FINAL'],color='orange', )
plt.legend(labels=['MEDIA_FINAL'], ncol=1, loc='upper left')

df_inicial.info(verbose=True, null_counts=True)

#Etapa de correlação de variáveis - Matriz de correlação
corr=df_inicial.corr()
sns.heatmap(corr,cmap= 'coolwarm')

print(corr['MEDIA_FINAL'].sort_values(ascending=False))

## Agora sabemos quais as perguntas que mais influenciam, então plotamos algumas informações
sns.countplot(x='Q006', hue='CLUSTER', data=df_inicial, palette='flare')
sns.countplot(x='Q024', hue='CLUSTER', data=df_inicial, palette='flare')
sns.countplot(x='Q008', hue='CLUSTER', data=df_inicial, palette='flare')
sns.countplot(x='Q010', hue='CLUSTER', data=df_inicial, palette='flare')
sns.countplot(x='Q002', hue='CLUSTER', data=df_inicial, palette='flare')

## Após a correlação, foi escolhido manter apenas as variáveis acima de 0,14 e abaixo de -0,22
df_inicial.drop(columns=['NOTA_FINAL','CLUSTER','CO_MUNICIPIO_RESIDENCIA','UF','MUNICIPIO','GINI','MED_ALUN_TURMA','TP_ANO_CONCLUIU','TP_SEXO','TP_NACIONALIDADE','Q011','NU_INSCRICAO','TP_ST_CONCLUSAO','TP_ESTADO_CIVIL','Q005'], inplace=True)

## ETAPA DE ML - Treinamento e Predição - 3 algoritmos (Regressão Linear, Random Forest e XGBoost)

#Separando as variáveis
x = df_inicial.drop(columns=['MEDIA_FINAL'])
y = df_inicial['MEDIA_FINAL']

X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=42)

#Coloca os valores em escalas
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##PRIMEIRO MODELO É REGRESSÃO LINEAR
start_time_rl = time.time()
from sklearn.linear_model import LinearRegression

rl_model = LinearRegression(n_jobs=6)

# rl_scores_r2 = cross_val_score(rl_model,X_train,y_train,cv=5,scoring='r2',verbose=3)
# rl_scores_mae = cross_val_score(rl_model,X_train,y_train,cv=5,scoring='neg_mean_absolute_error',verbose=3)
# rl_scores_rmse = cross_val_score(rl_model,X_train,y_train,cv=5,scoring='neg_root_mean_squared_error',verbose=3)

# print(rl_scores_r2)
# print(rl_scores_mae)
# print(rl_scores_rmse)

rl_model.fit(X_train, y_train)

#PREDIÇÃO
# rl_cv_pred = cross_val_predict(rl_model,X_train,y_train,cv=5)

# print('\n\nREGRESSÃO LINEAR CROSS VALIDATION')
# print('R2 score:', metrics.r2_score(y_train,rl_cv_pred))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, rl_cv_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, rl_cv_pred)))

rl_y_pred = rl_model.predict(X_test)

print('\n\nMETRICAS REGRESSÃO LINEAR')
print('R2 score:', metrics.r2_score(y_test,rl_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rl_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rl_y_pred)))

end_time_rl = time.time()
print("Tempo de Execução RL: %s segundos" % (end_time_rl - start_time_rl))

start_time_rf = time.time()
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(max_depth=20, max_features=12, min_samples_leaf=4,
                            min_samples_split=3, n_estimators=5000, random_state=42, n_jobs=6)

# rf_model = RandomForestRegressor(random_state=42)

# param_grid = [
#     {'max_depth': [20],
#       'max_features': [12],
#       'min_samples_leaf': [4],
#       'min_samples_split': [3],
#       'n_estimators': [5000],
#       }
# ]

# grid_search_rf = GridSearchCV(rf_model,param_grid,cv=2,verbose=2,n_jobs=10)
# grid_search_rf.fit(X_train,y_train)

# print(grid_search_rf.best_estimator_)

# rf_model = grid_search_rf.best_estimator_

# rf_scores_rmse = cross_val_score(rf_model,X_train,y_train,cv=2,scoring='neg_root_mean_squared_error',verbose=3,n_jobs=6)
# rf_scores_r2 = cross_val_score(rf_model,X_train,y_train,cv=2,scoring='r2',verbose=3,n_jobs=6)
# print(rf_scores_rmse)
# print(rf_scores_r2)

rf_model.fit(X_train, y_train)

#PREDIÇÃO
# rf_cv_pred = cross_val_predict(rf_model,X_train,y_train,cv=2,n_jobs=10)

# print('\n\nRANDOM FOREST CROSS VALIDATION')
# print('R2 score:', metrics.r2_score(y_train,rf_cv_pred))
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_cv_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, rf_cv_pred)))

rf_y_pred = rf_model.predict(X_test)

print('\n\nMETRICAS RANDOM FOREST')
print('R2 score:', metrics.r2_score(y_test,rf_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, rf_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_y_pred)))

end_time_rf = time.time()
print("Tempo de Execução RF: %s segundos" % (end_time_rf - start_time_rf))

## TERCEIRO MODELO É O XGBOOST
start_time_xgb = time.time()
import xgboost as xgb

# params = {
#         'min_child_weight': [5],
#         'max_depth': [10],
#         'subsample': [1.0],
#         'colsample_bytree': [0.7],
#         'n_estimators': [5000],
#         'learning_rate': [0.05],
#         'gamma': [0.1],
#         'random_state': [42],
#         # usando GPU
#         'tree_method': ['gpu_hist'],
#         'predictor': ['gpu_predictor'],
#         'eval_metric': ['rmse']
# }

# xgb_model = xgb.XGBRegressor(random_state=42)

# grid_search_xgb = GridSearchCV(xgb_model,params,cv=2,verbose=3, scoring='neg_root_mean_squared_error',n_jobs=6)
# grid_search_xgb.fit(X_train, y_train)

# print(grid_search_xgb.best_score_)
# print(grid_search_xgb.best_params_)

# # #Melhores parâmetros
# xgb_model = grid_search_xgb.best_estimator_

#CROSS VALIDATION
# cv_results = xgb.cv(
#     grid_search_xgb.best_params_,
#     dtrain,
#     seed=42,
#     nfold=5,
#     metrics={'rmse'},
#     early_stopping_rounds=50
# )

# print(cv_results)
# print(cv_results['test-rmse-mean'].min())

params = {'colsample_bytree': 0.7, 'eval_metric': 'rmse', 'gamma': 0.3, 'learning_rate': 0.05, 'max_depth': 10, 'min_child_weight': 5, 'n_estimators': 100, 'subsample': 1.0, 'n_jobs': 6}

xgb_model = xgb.XGBRegressor(**params)
xgb_model.fit(X_train, y_train)

#PREDIÇÃO
xgb_y_pred = xgb_model.predict(X_test)

print('\n\nMETRICAS XGBOOST')
print('R2 score:', metrics.r2_score(y_test,xgb_y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, xgb_y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, xgb_y_pred)))

end_time_xgb = time.time()
print("Tempo de Execução XGBoost: %s segundos" % (end_time_xgb - start_time_xgb))

print("Tempo de Execução RL: %s segundos" % (end_time_rl - start_time_rl))
print("Tempo de Execução RF: %s segundos" % (end_time_rf - start_time_rf))
print("Tempo de Execução XGBoost: %s segundos" % (end_time_xgb - start_time_xgb))

print("Tempo de Execução Total: %s segundos" % (time.time() - start_time))
