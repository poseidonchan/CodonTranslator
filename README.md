# CodonTranslator

A conditional codon sequence optimizer based on sequence and protein prefixes.

## Installation

### Requirements

- PyTorch and transformers
- ESM is required at runtime for protein prefixing. Please note that the ESM package has conflicts with the newest transformers library, please install `esm` at first and then install newest `transformers`.

```bash
pip install -e ./
```


## Pretrained model
The final checkpoint is available at [google drive](https://drive.google.com/drive/folders/1ekUkUzlhqSWra0SioagAsR_pdZiy22NS?usp=drive_link), please download the whole folder and specify the folder path for use.

## Usage examples
### Basic usage
```python
from CodonTranslator import CodonTranslator

model = CodonTranslator.from_pretrained(
    model_path="/final_model", # model.safetensors, vocab.json, trainer_config.json
    device="cuda"
)

dna = model.sampling(
    species="Homo sapiens",
    protein_seq="MSEQUENCEA",
    enforce_mapping=True,
    temperature=1,
    top_k=50,
    top_p=1,
)
print(dna)
```
### Batch inference
```python
seqs = model.batch_inference(
    species=["Homo sapiens", "Homo sapiens"],
    protein_seqs=["MSEQUENCEA", "MSEQUENCEA"],
    enforce_mapping=True,
    temperature=1,
    top_k=50,
    top_p=1,
)
print(seqs)
```

## Issues
Please feel free to raise issues and ask questions. The pretraining code will be released once the paper is accepted. The pretraining dataset can be shared upon user's request. Please contact me via [email](cys@umd.edu) for the dataset sharing.
