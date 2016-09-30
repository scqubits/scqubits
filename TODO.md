# TODO

Some ideas for things to implement:

- Update all plot commands to an OO interface and (as an option?) return the plot's figure. This will allow for much more flexibility when adjusting plots (as in SpectrumData.plot())
- For zero-pi might want to get rid of ECS (or alternatively one of the other inter-dependnt parameters) as a parameter to the constructor, and instead calculate it based on the EC and ECJ. 
- Get_spectrum_vs_paramvals() currently may give inconsistent results when parameter is one of the inter-dependent ones (eg: EC, ECJ, ECS), might want to raise exceptions, or update other ones accordingly. 
- Make showing of the progress_bar in some methods optional; sometimes when running things in parallel, all show up at once. 
- Add an update(d) method, that updates the parameters from a dictionary d

OPTIONAL:
- Could get rid of _eigenvals_stored concept; make each qubit object just a "config" that knows its Hamiltonian and how to diagonalize itself. We might want to store actual data elsewhere (i.e. "no side effects").



