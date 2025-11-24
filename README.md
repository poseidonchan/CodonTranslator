CodonTranslator
=========================

A conditional codon sequence optimizer based on sequence and protein prefixes.

Installation
-----------------------------------

pip install -e ./

Requirements
------------

- PyTorch and transformers
- ESM is required at runtime for protein prefixing. Please note that the ESM package has conflicts with the newest transformers library, please install `esm` at first and then install newest `transformers`.

## Pretrained model
The final checkpoint is available at [google drive](https://drive.google.com/drive/folders/1ekUkUzlhqSWra0SioagAsR_pdZiy22NS?usp=drive_link), please download the whole folder and specify the folder path for use.

Basic usage
-----------
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
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(dna)
```
Batch inference
---------------
```python
seqs = model.batch_inference(
    species=["Homo sapiens", "Homo sapiens"],
    protein_seqs=["MSEQUENCEA", "MSEQUENCEA"],
    enforce_mapping=True,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(seqs)
```

