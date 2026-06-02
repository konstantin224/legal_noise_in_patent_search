import numpy as np
from config_py import *

delta_avg = np.load(NAME_DELTA_AVG)

alpha = 1
def transform_query_vector(vector): # делаем линейно-аддитивное преобразование вектора запроса
    return vector + alpha * delta_avg