# settings.py

from sc_qubits.utils.constants import FileType

file_format = FileType.h5   # choose FileType.csv instead for generation of comma-separated values files

# switch for display of progress bar; default: show only in ipython
try:
    __IPYTHON__
    progressbar_enabled = True
except:
    progressbar_enabled = False

