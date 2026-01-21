# Mitigating Social Bias in LLMs via KnowBias



## Overview

**KnowBias**, a lightweight and conceptually distinct framework that mitigates bias by strengthening, rather than suppressing, neurons encoding "knowing bias".


## Finding know-bias neurons

1. [`run_.py`](run_.py) will calculate the attribution scores of each neuron and each layer for the given LLM, and save into a .npy file; data can be found in 📂 data/ with constructed simple and abstract-level questions.
2. [`analysis_res.py`](analysis_res.py) will based on the .npy file obtained from the previous step, and output all the neuron information, like (1,2) means the neuron for index 2 in layer 1.

## Debiasing evaluation
Run [`evaluation_bias_all.py`](evaluation_bias_all.py) to evaluate the debiasing performance and general capabilities for the LLM by enhancing these know-bias neurons `λ`.


[//]: # (## Citation)

[//]: # (```bibtex)

[//]: # ()
[//]: # (```)
