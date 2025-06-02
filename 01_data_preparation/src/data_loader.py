import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Función para interpolar los datos del dataframe
def imputar_datos_consumo(df, columnas_consumo):
    """
    Imputa valores faltantes en columnas específicas de un DataFrame.

    Esta función procesa un DataFrame creando una copia para no modificar el original,
    y para cada columna especificada:
        - Establece en 0 todos los valores antes de la primera observación válida (no NaN).
        - Interpola linealmente los valores faltantes a partir de la primera observación válida.
        - Si una columna está completamente vacía (solo NaN), la rellena con 0.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame de entrada con los datos a imputar. Generalmente indexado por fechas u otro índice secuencial.
    columnas_consumo : list de str
        Lista de nombres de columnas del DataFrame donde se aplicará el proceso de imputación.

    Retorna:
    --------
    pandas.DataFrame
        Un nuevo DataFrame con los valores imputados en las columnas especificadas.
    """
    # Crear una copia para no modificar el DataFrame original
    df_imputado = df.copy()

    # Iterar por cada columna de consumo
    for columna in columnas_consumo:
        # Encontrar el índice de la primera observación válida (no NaN)
        primer_indice_valido = df_imputado[columna].first_valid_index()

        if primer_indice_valido is not None:
            # Si hay al menos un valor válido en la columna:
            # a) Establecer en 0 todos los valores ANTES de la primera observación válida
            df_imputado.loc[:primer_indice_valido, columna] = df_imputado.loc[:primer_indice_valido, columna].fillna(0)

            # b) Interpolar los valores faltantes a partir de la primera observación válida
            df_imputado.loc[primer_indice_valido:, columna] = df_imputado.loc[primer_indice_valido:, columna].interpolate(
                method='linear', limit_direction='both'
            )
        else:
            # Si toda la columna es NaN, rellenar con 0
            df_imputado[columna] = df_imputado[columna].fillna(0)

    return df_imputado

def crear_indice_fecha(df):
    """
    Crea un índice de fecha para un DataFrame.

    Esta función convierte la columna 'Datetime' de un DataFrame en un índice de tipo datetime,
    asegurando que el índice sea único y ordenado.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene una columna 'Datetime' que se convertirá en el índice.

    Retorna:
    --------
    pandas.DataFrame
        Un nuevo DataFrame con la columna 'Datetime' como índice datetime.
    """
    # Se combinan las columnas 'Year' y 'Month' para crear una nueva columna 'Datetime' si no existe
    if 'Datetime' not in df.columns:
        # Comprobando si las columnas 'Year' y 'Month' existen en el DataFrame
        if 'Year' in df.columns and 'Month' in df.columns:
            # Combinando columnas de Year y Month para crear una nueva columna 'Datetime'
            df['Datetime'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str), format='%Y-%m')
            # eliminando las columnas 'Year' y 'Month'
            df = df.drop(['Year', 'Month'], axis=1)  # Eliminando las columnas 'Year' y 'Month'
    return df


def mix_heatmap(df, columnas_consumo):
    """
    Crea un heatmap de consumo de energía.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un heatmap
    que muestra el consumo de energía a lo largo del tiempo.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el heatmap generado.
    """
    # Obtenemos los años del índice
    years = df.index.year

    # Creamos un DataFrame agrupado por año
    df_annual = df.groupby(years)[columnas_consumo].sum()

    # Calculamos los porcentajes por año
    for year in df_annual.index:
        total = df_annual.loc[year].sum()
        if total > 0:  # Evitamos divisiones por cero
            df_annual.loc[year] = df_annual.loc[year] / total * 100

    # Transponemos para tener fuentes en filas y años en columnas
    df_annual_transposed = df_annual.T

    # Visualización: Heatmap por Año (con selección de años para mayor claridad)

    plt.figure(figsize=(16, 10))

    # Seleccionamos un subconjunto de años para evitar sobrecarga visual, por ejemplo, uno de cada 5 años
    all_years = sorted(list(set(years)))
    selected_years = all_years[::5]  # Toma uno cada 5 años
    if all_years[-1] not in selected_years:  # Asegura incluir el año más reciente
        selected_years.append(all_years[-1])

    # Filtramos el DataFrame para mostrar solo los años seleccionados
    df_annual_selected = df_annual_transposed[selected_years]

    # Creamos el heatmap
    sns.heatmap(df_annual_selected, cmap='viridis', 
                cbar_kws={'label': 'Porcentaje de Consumo (%)'}, 
                linewidths=0.3,
                annot=True,  # Mostramos los valores
                fmt='.1f')   # Con un decimal

    plt.title('Evolución del Consumo Energético por Fuente (Años Seleccionados)', fontsize=16, fontweight='bold')
    plt.xlabel('Año', fontsize=14)
    plt.ylabel('Fuente de Energía', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sector_consumo(df, columnas_consumo):
    """
    Crea un gráfico de lineas del consumo por sector.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un gráfico
    de lineas que muestra el consumo de energías renovables a lo largo de los años para cada sector.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el gráfico generado.
    """
   # Se calcula el total de energía renovable consumida por todos los sectores
    df_processed_4 = df.groupby(['Datetime', 'Sector'])[columnas_consumo].sum()
    df_processed_4['Total Renewable Energy'] = df_processed_4.sum(axis=1)
    df_processed_4 = df_processed_4.reset_index('Sector')

    # Grafico de líneas por sector
    plt.figure(figsize=(14, 7))
    df_processed_1 = df_processed_4['Total Renewable Energy']
    df_processed_1[df_processed_4['Sector'] == 'Commercial'].plot(kind = 'line', linewidth=2, label='Commercial')
    df_processed_1[df_processed_4['Sector'] == 'Residential'].plot(kind = 'line', linewidth=2, label='Residential')
    df_processed_1[df_processed_4['Sector'] == 'Industrial'].plot(kind = 'line', linewidth=2, label='Industrial')
    df_processed_1[df_processed_4['Sector'] == 'Transportation'].plot(kind = 'line', linewidth=2, label='Transportation')
    df_processed_1[df_processed_4['Sector'] == 'Electric Power'].plot(kind = 'line', linewidth=2, label='Electric Power')
    plt.title('Consumo Total por mes de Energía Renovable en EE.UU por Sector. (1973-2024)', fontsize=16)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Consumo (Trillion BTU)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Sector', fontsize=10)
    plt.tight_layout()
    plt.show()

def bar_sector_consumo(df, columnas_consumo):
    """
    Crea un gráfico de barras del consumo por sector.

    Esta función toma un DataFrame y una lista de columnas de consumo, y genera un gráfico
    de barras que muestra el consumo de energías renovables a lo largo de los años para cada sector.

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame que contiene los datos de consumo de energía.
    columnas_consumo : list de str
        Lista de nombres de columnas que representan diferentes tipos de consumo.

    Retorna:
    --------
    matplotlib.figure.Figure
        Un objeto Figure que contiene el gráfico generado.
    """
    # Se calcula el total de energía renovable consumida por todos los sectores
    df_processed_5 = df.groupby(['Datetime', 'Sector'])[columnas_consumo].sum()
    df_processed_5['Total Renewable Energy'] = df_processed_5.sum(axis=1)
    df_processed_5 = df_processed_5.reset_index('Sector')

    # Obtenemos los años del índice
    years = df.index.year

    # Creamos un DataFrame agrupado por año y sector
    df_processed_5 = df_processed_5.groupby([years,'Sector'])['Total Renewable Energy'].sum().unstack()

    # Creamos el gráfico de barras apiladas
    plt.figure(figsize=(16, 10))
    df_processed_5.plot(kind='bar', stacked=True, colormap='viridis', edgecolor='black')
    plt.title('Consumo Total por año de Energía Renovable por Sector (1973-2024)', fontsize=16)
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Consumo (Trillion BTU)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Sector', fontsize=10)
    plt.tight_layout()
    plt.show()