# ERP-Transformer-vs-NonTransformer
Code and materials for my MSc Data Science Extended Research Project (University of Manchester)
This repo contains the code used in my MSc Data Science project at the University of Manchester.  
The goal was to compare several Transformer (LLaMA2, Mistral) and non-Transformer (RWKV, StripedHyena) models on neural encoding tasks.

Requirements:
- Python 3.8.20
- PyTorch 2.4.1  
- HuggingFace Transformers  
- NumPy / SciPy / Pandas  / flash_attn

Data Availability：
fMRI data: from Pereira et al. (2018), available at https://doi.org/10.1038/s41467-018-03068-4
Scripts will output .mat files containing embeddings and results.
These are not included in the repo due to size.

References
Pereira, F. et al. (2018). Toward a universal decoder of linguistic meaning from brain activation. Nature Communications.
