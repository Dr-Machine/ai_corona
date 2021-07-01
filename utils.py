import numpy as np


def window(img: np.ndarray) -> np.ndarray:
    """
    Calculates the window.

    Args:
        img (np.ndarray): Image for which the window will be calculated.

    Returns:
        np.ndarray: Calculated window.
    """
    WL, WW = -600, 1500
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / (np.max(X) / 255.0)
    X = X.astype('uint8')
    X = np.expand_dims(X, axis=-1)
    X = np.concatenate([X, X, X], axis=-1)
    return X


def generate_report(diagnosis: dict) -> None:
    """
    Generates and prints a nice-looking report.

    Args:
        diagnosis (dict): Diagnosis dictionary.
    """
    normal, pneumonia, covid = diagnosis['n'], diagnosis['p'], diagnosis['c']
    normal = round(float(normal), 2)
    pneumonia = round(float(pneumonia), 2)
    covid = round(float(covid), 2)
    report = ('---------------------------------------\n'
              '            Diagnosis Report           \n'
              '---------------------------------------\n'
              f'Normal:                         {normal}% \n'
              f'Pneumonia (non COVID-19):       {pneumonia}% \n'
              f'COVID-19:                       {covid}% \n'
              '---------------------------------------')
    print(report)
