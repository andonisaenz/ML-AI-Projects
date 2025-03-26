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




















