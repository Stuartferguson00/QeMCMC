











# Quantum-enhanced Markov Chain Monte Carlo

This is a code designed to impliment the algorithm from the paper: It builds upon the numerics in Layden's work on the [Quantum enhanced Markov Chain Monte Carlo (QeMCMC)](https://www.nature.com/articles/s41586-023-06095-4).

Please see also https://github.com/pafloxy/quMCMC, which was used as a starting point for this code, and https://github.com/Stuartferguson00/CGQeMCMC which impliments a more complex algorithm, allowing a restricted number of qubits.

## Authors and contact
For questions and suggestions, please contact Stuart Ferguson: S.A.Ferguson-3@sms.ed.ac.uk

This project was created by:
* Stuart Ferguson


## License

Distributed under the MIT License. See LICENSE.txt for more information.


## Acknowledgements

* https://github.com/pafloxy/quMCMC, an open-source code which was used as a starting point for this code.
* [QeMCMC by David Layden et al.](https://www.nature.com/articles/s41586-023-06095-4)
* [Quantum-enhanced Markov Chain Monte Carlo for systems larger than your Quantum Computer by S. Ferguson and P. Wallden](https://arxiv.org/abs/2405.04247)
* [Qulacs Simulator](https://quantum-journal.org/papers/q-2021-10-06-559/)
Quantum-enhanced Markov chain Monte Carlo by David Layden et al. https://www.nature.com/articles/s41586-023-06095-4



## Tutorial


We also include a tutorial jupyter notebook "tutorial.ipynb", where a Markov Chain Monte Carlo algorithm is run for an example 9 spin instance. Classical "Uniform" and "local" update proposals are compared with a QeMCMC implimentation using only 3 simulated qubits.


