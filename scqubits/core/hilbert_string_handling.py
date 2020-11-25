from typing import Any, Callable, Dict, Iterator, Optional, Tuple, List, Union
import numpy as np
import qutip as qt
from numpy import ndarray
from qutip.qobj import Qobj

qutip_dict = {
    'cos': 'Qobj.cosm',
    'dag': 'Qobj.dag',
    'conj': 'Qobj.conj',
    'exp': 'Qobj.expm',
    'sin': 'Qobj.sinm',
    'sqrt': 'Qobj.sqrtm',
    'trans': 'Qobj.trans'
}

matrix_one = qt.qeye(10)
matrix_two = qt.qeye(10) * 3


def replace_string(string: str):
    for item, value in qutip_dict.items():
        if item in string:
            string = string.replace(item, value)
    return string


def run_string_code(string: str, variable: str):
    string = replace_string(string)
    answer = eval(string)
    return answer


fun_string = 'cos(sqrt(matrix_one) + matrix_two)'
run_string_code(fun_string)
