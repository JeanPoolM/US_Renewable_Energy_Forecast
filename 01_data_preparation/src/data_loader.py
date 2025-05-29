import pandas as pd
import numpy as np

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