# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 13:17:47 2025

@author: Andoni Sáenz
"""

#%% IMPORTAR LIBRERÍAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% CARGAR LOS DATOS

df = pd.read_csv('AbandonoEmpleados.csv', sep = ';', index_col = 'id', na_values = "#N/D")
print(df.info())
print(df.describe())
print(df.shape)

#%% ANÁLISIS DE VALORES NULOS

print(df.isna().sum().sort_values(ascending = False)) 
# Variables: anos_en_puesto, conciliacion tienen 1000 NAs --> Eliminar variables
# Variables: sexo, educacion, satisfaccion_trabajo, implicacion --> Se imputan valores tras EDA

# Se eliminan variables con más de 1000 NAs
df.drop(columns = ['anos_en_puesto', 'conciliacion'], inplace = True)

#%% EXPLORATORY DATA ANALYSIS (EDA) - Variables categóricas

# Seleccionar variables categóricas
cat = df.select_dtypes('O')

# Crear función para graficar variables
def graficos_categoricos(cat):
    
    # Definir gráfico
    fig, ax = plt.subplots(nrows = 7, ncols = 2, figsize = (16, 50))
    
    # Aplanar gráfico para definirlo como de 1 dimensión en vez de 2
    ax = ax.flat
    
    # Añadir gráficos
    for n, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax = ax[n])
        ax[n].set_title(variable, fontsize = 8, fontweight = 'bold')
        ax[n].tick_params(labelsize = 8)
        
graficos_categoricos(cat)

# Variable mayor_edad solo tiene 1 valor --> eliminarla
# Imputaciones pendientes
    # educacion --> imputar por 'Universitaria' 
    # satisfaccion_trabajo --> imputar por 'Alta'
    # implicacion --> imputar por 'Alta'
    
df.drop(columns = 'mayor_edad', inplace = True)
df['educacion'] = df['educacion'].fillna('Universitaria')
df['satisfaccion_trabajo'] = df['satisfaccion_trabajo'].fillna('Alta')
df['implicacion'] = df['implicacion'].fillna('Alta')


#%% EXPLORATORY DATA ANALYSIS (EDA) - Variables numéricas

# Seleccionar variables numéricas
num = df.select_dtypes('number')

# Crear función para obtener estadísticos
def estadisticos_cont(num):
    est = num.describe().T 
    
    # Añadir la mediana
    est['median'] = num.median()
    
    # Mover la mediana al lado de la media
    est = est.iloc[:, [0,1,8,2,3,4,5,6,7]]
    
    return est

print(estadisticos_cont(num))

# Variable empleados solo tiene 1 valor --> eliminarla
# Variable sexo tiene 4 valores --> eliminarla
# Variable horas_quincena solo tiene 1 valor --> eliminarla
# De los valores nulos pendientes de imputar solo queda la variable sexo, pero como se va a eliminar no se imputa

df.drop(columns = ['empleados', 'sexo', 'horas_quincena'], inplace = True)


#%% GENERACIÓN DE INSIGHTS

### CUANTIFICAR TASA DE ABANDONO

print(df['abandono'].value_counts(normalize = True).round(4)*100)

### PERFIL DEL EMPLEADO QUE ABANDONA LA EMPRESA

# Transformar variable abandono a numérica
df['abandono'] = df['abandono'].map({'No': 0, 'Yes': 1})

### Abandono por nivel educativo
educ = df.groupby('educacion')['abandono'].mean().sort_values(ascending = False)*100
plt.bar(educ.index, educ)
plt.show()

### Abandono por estado civil
civ = df.groupby('estado_civil')['abandono'].mean().sort_values(ascending = False)*100
plt.bar(civ.index, civ)
plt.show()

### Análisis por horas extras
ext = df.groupby('horas_extra')['abandono'].mean().sort_values(ascending = False)*100
plt.bar(ext.index, ext)
plt.show()

### Abandono por puesto
puesto = df.groupby('puesto')['abandono'].mean().sort_values(ascending = False)*100
plt.bar(puesto.index, puesto)
plt.xticks(rotation = 45)

### Análisis por salario mensual
df.groupby('abandono')['salario_mes'].mean().plot.bar()

# Perfil medio del empleado que ha abandonado la empresa:
    # Bajo nivel educativo
    # Soltero
    # Trabaja en ventas
    # Bajo salario
    # Alta carga de horas extras
    
### CUANTIFICAR EL IMPACTO ECONÓMICO DEL PROBLEMA

# Coste de fuga de los empleados según el estudio "Cost of Turnover" del Center for American Progress
    # Empleados que ganan menos de 30000$ --> 16.1% de su salario
    # Empleados que ganan entre 30000$-50000$ --> 19.7% de su salario
    # Empleados que ganan entre 50000$-75000$ --> 20.4% de su salario
    # Empleados que ganan más de 75000$ --> 21% de su salario
    
# Crear variable salario_anual del empleado
df['salario_anual'] = df['salario_mes'].transform(lambda x: x*12)

# Crear lista de condiciones
condiciones = [(df['salario_anual']  <= 30000), 
               (df['salario_anual'] > 30000) & (df['salario_anual'] <= 50000),
               (df['salario_anual'] > 50000) & (df['salario_anual'] <= 75000),
               (df['salario_anual'] > 75000)]

# Calcular el impacto de cada empleado
resultados = [df['salario_anual']*0.161, df['salario_anual']*0.197, 
              df['salario_anual']*0.204, df['salario_anual']*0.210]

# Aplicar select sobre el DataFrame 
df['impacto_abandono'] = np.select(condiciones, resultados, default = -99)
print(df)

### COSTE ECONÓMICO DEL ABANDONO DURANTE EL ÚLTIMO AÑO

coste_ult_año = df.loc[df['abandono'] == 1]['impacto_abandono'].sum()
print(coste_ult_año)

### CUANTO CUESTA QUE LOS EMPLEADOS NO ESTÉN MOTIVADOS (IMPLICACIÓN BAJA)?

coste_impl_baja = df.loc[(df['abandono'] == 1) & (df['implicacion'] == 'Baja')]['impacto_abandono'].sum()
print(coste_impl_baja)

### CUANTO DINERO SE PODRÍA AHORRAR LA EMPRESA FIDELIZANDO MEJOR A LOS EMPLEADOS?

print(f"Reducir un 10% la fuga de empleados nos ahorraría un {int(coste_ult_año * 0.1)}$ al año.")
print(f"Reducir un 20% la fuga de empleados nos ahorraría un {int(coste_ult_año * 0.2)}$ al año.")
print(f"Reducir un 30% la fuga de empleados nos ahorraría un {int(coste_ult_año * 0.3)}$ al año.")

### TRAZAR ESTRATEGIAS ASOCIADAS A INSIGHTS DE ABANDONO

# Puesto con mayor tasa de abandono: representantes de ventas
# Calcular proporción de representantes de ventas que se han ido el último año

tot_repr_ventas = len(df.loc[df['puesto'] == 'Sales Representative'])
tot_repr_ventas_abandono = len(df.loc[(df['puesto'] == 'Sales Representative') & (df['abandono'] == 1)])
prop_abandono_ventas = tot_repr_ventas_abandono/tot_repr_ventas
print(round(prop_abandono_ventas, 2))

# Alrededor de un 40% de representantes de ventas abandonaron su puesto el último año

# Calcular número de representantes de ventas que se podrían ir este año 
tot_repr_ventas_actual = len(df.loc[(df['puesto'] == 'Sales Representative') & (df['abandono'] == 0)])
prevision_abandono_ventas = int(tot_repr_ventas_actual*prop_abandono_ventas)
print(prevision_abandono_ventas)

# De los actuales representantes de ventas, 19 se podrían marchar durante este año

# Tiene sentido aplicar un plan específico para ellos? 
# Cuánto podría ahorrar la empresa si disminuimos la fuga un 30%?

# Sobre los representantes de ventas que se podrían marchar cuántos podemos retener? (Hipótesis 30%)
# Cuánto dinero puede suponer?

reten_empleados_ventas = int(prevision_abandono_ventas*0.3)
ahorro_reten_empleados_ventas = round(df.loc[(df['puesto'] == 'Sales Representative') & (df['abandono'] == 0),
                                       'impacto_abandono'].sum() * prop_abandono_ventas * 0.3, 2)

print(f'Podemos retener {reten_empleados_ventas} representantes de ventas, suponiendo un ahorro de {ahorro_reten_empleados_ventas}$.')

# Se podrían llegar a ahorrar más de 37000$, por lo que la empresa podría invertir hasta esa cantidad en acciones para retener a representantes de ventas.
# Dichas acciones se estarían pagando solas con la pérdida evitada.

#%% CREACIÓN DE UN MODELO DE MACHINE LEARNING

df_ml = df.copy()
print(df_ml.info())

### PREPARACIÓN DE DATOS PARA LA MODELIZACIÓN

### Transformar variables categóricas a numéricas

from sklearn.preprocessing import OneHotEncoder

# Seleccionar variables categóricas
cats = df_ml.select_dtypes('O')

# Instanciar variables categóricas
ohe = OneHotEncoder(sparse_output = False)

# Entrenar con variables categóricas seleccionadas
ohe.fit(cats)

# Aplicar a variables categóricas seleccionadas
cats_ohe = ohe.transform(cats)

# Nombrar variables categóricas
cats_ohe = pd.DataFrame(cats_ohe, columns = ohe.get_feature_names_out(input_features = cats.columns)).reset_index(drop = True)

print(cats_ohe)

### Juntar variables numéricas y categóricas modificadas

nums = df.select_dtypes('number').reset_index(drop = True)

df_ml = pd.concat([cats_ohe, nums], axis = 1)
print(df_ml)
 
#%% DISEÑO DEL MODELO BASE - REGRESIÓN LOGÍSTICA

# Separar variables predictoras y target
X = df_ml.drop(columns = 'abandono')
y = df_ml['abandono']

# Seoarar los datos de train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 25)

# Escalar datos antes de entrenar el modelo

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Creación del modelo de Regresión Logística
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter = 500, random_state = 17, class_weight = 'balanced')

# Entrenar modelo base de Regresión Logística
logreg.fit(X_train_scaled, y_train)

### Predicción y validación sobre el test
pred_lg = logreg.predict_proba(X_test_scaled)[:, 1]

### Evaluación del modelo base
from sklearn.metrics import roc_auc_score

print(f'El modelo de Regresión Logística permitió obtener un ROC-AUC score de {roc_auc_score(y_test, pred_lg): .3f}.')

#%% DISEÑO DEL MODELO ALTERNATIVO - DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Creación del árbol de decisión

dt = DecisionTreeClassifier(random_state = 25, class_weight = 'balanced')

### FINE-TUNING DE HIPERPARÁMETROS

pg = {'max_depth': [5, 10, 20], 
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]}

dt = GridSearchCV(estimator = dt, param_grid = pg, cv = 5, n_jobs = -1)

# Entrenar el Árbol de Decisión

dt = dt.fit(X_train_scaled, y_train)

print(f'Los parámetros óptimos para el Árbol de Decisión son: {dt.best_params_}')

### Predicción y validación sobre el test

dt = dt.best_estimator_
pred_dt = dt.predict(X_test_scaled)

### Evaluar el modelo alternativo

print(f'El Decision Tree Classifier permitió obtener un ROC-AUC score de {roc_auc_score(y_test, pred_dt): .3f}.')

#%% ELECCIÓN DEL MODELO FINAL

# Se selecciona el modelo de Regresión Logística tras haber obtenido un mayor ROC-AUC score

# Evaluar el peso de cada variable en la predicción final

logreg_coef = logreg.coef_[0]
peso_vars = pd.DataFrame({'variable': X.columns, 'importancia': logreg_coef})
peso_vars = peso_vars.reindex(peso_vars['importancia'].abs().sort_values(ascending = False).index)

plt.figure(figsize= (24,18))
plt.barh(peso_vars['variable'], peso_vars['importancia'].abs())
plt.xlabel('Importancia')
plt.ylabel('Variable')
plt.title('Importancia de las variables en el modelo de Regresión Logística')
plt.show()

# Incorporación del scoring al dataframe principal

df['scoring_abandono'] = logreg.predict_proba(df_ml.drop(columns = 'abandono'))[:, 1]
print(df)

from sklearn.metrics import classification_report

print(classification_report(y_test, logreg.predict(X_test_scaled)))










