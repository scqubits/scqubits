# TODO

Some ideas for things to implement:

- Update all plot commands to an OO interface and (as an option?) return the plot's axes. This will allow for much more flexibility when customizing plots. I.e. the user
  should be able to pass an axes to a plotting command (say of a wavefunction or potential), and in that case those very axes should be used for plotting. 
    - Some of the zero_pi methods are done now... but need to update fluxonium, potential plots, etc.
- For zero-pi might want to get rid of ECS (or alternatively one of the other inter-dependnt parameters) as a parameter to the constructor, and instead calculate it based on the EC and ECJ. 
- Get_spectrum_vs_paramvals() currently may give inconsistent results when parameter is one of the inter-dependent ones (eg: EC, ECJ, ECS), might want to raise exceptions, or update other ones accordingly. 
- Make showing of the progress_bar in some methods optional; sometimes when running things in parallel, all show up at once. 
- Add an update(d) method, that updates the parameters from a dictionary d

OPTIONAL:
- Could get rid of _eigenvals_stored concept; make each qubit object just a "config" that knows its Hamiltonian and how to diagonalize itself. We might want to store actual data elsewhere (i.e. "no side effects").



