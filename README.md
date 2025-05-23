# transition-metal-orbital-classifier

The goals here is to teach a large vision model such as GPT-4o to recognize the shape of molecular orbitals. The approach being taken here is to teach them how to identiy atomic orbitals first, and then the shapes and symmetries of a few basis molecular orbitals. Since one can fine-tune, fine-tuned models, and sequential approach is being professed here.

Classifies atomic orbitals on first-row transition metals by:
- Running ROHF calculations (e.g., Cr, Mn, Ni, Fe)
- Performing meta-Löwdin AO analysis
- Identifying orbitals with high 3d character
- Optionally leveraging GPT to produce alternative orbital labels
- Generating `.cube` files for 3d orbitals

## Features
- **Configurable systems**: Add or edit transition metal atoms, spins, charges in one dictionary.
- **Automated labeling**: Threshold-based classification of orbitals by fraction of 3d character.
- **Cube file generation**: Creates `.cube` files for orbitals with fraction of 3d ≥ 75%.
- **Optional GPT integration**: Compares GPT-derived labels to threshold-based labels in unittests.

## Requirements
- Python 3.7+
- [PySCF](https://pyscf.org) for electronic structure calculations
- [OpenAI Python library](https://github.com/openai/openai) (optional, for GPT-based labeling)
- [python-dotenv](https://github.com/theskumar/python-dotenv) (optional, for loading `OPENAI_API_KEY`)

Install with:
```bash
pip install pyscf openai python-dotenv
# transition-metal-orbital-classifier
Classifies atomic orbitals on first row transition metals. 
