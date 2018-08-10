# sc_qubits
Python library for superconducting qubits

`sc_qubits2.py` provides basic routines for the superconducting qubits we use most frequently in our group (at least right now, may want to add more as time goes on):
- transmon/CPB
- fluxonium
- variants of the 0-pi qubit (NOTE: the disorder definitions used in the latest versions of the code are from the coherence paper)

`operators.py` common operators (returned in array or sparse format)

`config.py` collects all global variables and settings in a central place

`plotting.py` comprises all plotting routines which are not qubit specific and hence of general use

`testing.py` is to be used with `nosetests`. It's not complete, but a start for checking whether future commits break anything.

`testing-sc_qubits.ipynb` is an ipython notebook demonstrating a lot of the existing parts of the library.

`Fluxonium-v2.ipynb` demonstrates some additional parts that we have used specifically for the fluxonium project, but are more generally applicable.


The code is placed here not only for easy sharing, but also for working together to improve and extend it. (This works in the usual github way: create your own branch, make changes to your branch, commit changes, and when finished place a pull request so the group can take a look, comment on your changes if needed, and then merge your changes back into the master branch.)


