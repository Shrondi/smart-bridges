def is_anomalous(df, min_duration=2):
    """
    Procesa un DataFrame para verificar si es anómalo.
    Comprueba si los timestamps están desordenados o si la duración es menor a 2 segundos.

    Args:
        df (pd.DataFrame): DataFrame con una columna 'timestamp' como índice o columna.

    Returns:
        bool: True si el DataFrame es anómalo, False en caso contrario.
    """

    # Verificar si los timestamps están ordenados
    if not df.index.is_monotonic_increasing:
        return True  # Es anómalo si los timestamps no están ordenados

    # Calcular la duración entre el primer y el último timestamp
    primera_ts = df.index[0]
    ultima_ts = df.index[-1]
    duracion = (ultima_ts - primera_ts).total_seconds()

    # Verificar si la duración es menor a min_duration
    if duracion < min_duration:
        return True  # Es anómalo si la duración es menor a min_duration

    return False  # No es anómalo