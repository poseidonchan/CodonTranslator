# Minimal copy of CodonTokenizer from src/tokenizer.py to keep the package self-contained.
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass(frozen=True)
class SpecialIds:
    pad: int = 0
    unk: int = 1
    bos: int = 2
    eos: int = 3

    def to_dict(self) -> Dict[str, int]:
        return {"pad": self.pad, "unk": self.unk, "bos": self.bos, "eos": self.eos}


class CodonTokenizer:
    __slots__ = (
        "codons",
        "_special_token_str",
        "vocab",
        "ids_to_tokens",
        "_special_ids",
        "_num_special_tokens",
        "_genetic_code",
        "_codon2aa_char",
        "_aa2codons_char",
    )

    def __init__(
        self,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<stop>",
        **_: Any,
    ) -> None:
        bases = ("A", "C", "G", "T")
        self.codons: List[str] = [a + b + c for a in bases for b in bases for c in bases]

        special_tokens = [pad_token, unk_token, bos_token, eos_token]
        self._special_token_str = {"pad": pad_token, "unk": unk_token, "bos": bos_token, "eos": eos_token}

        self.vocab: Dict[str, int] = {}
        for i, tok in enumerate(special_tokens):
            self.vocab[tok] = i
        for codon in self.codons:
            self.vocab[codon] = len(special_tokens) + (len(self.vocab) - len(special_tokens))

        self.ids_to_tokens: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        self._special_ids = SpecialIds(
            pad=self.vocab[pad_token],
            unk=self.vocab[unk_token],
            bos=self.vocab[bos_token],
            eos=self.vocab[eos_token],
        )
        self._num_special_tokens = len(special_tokens)

        self._genetic_code: Dict[str, str] = {
            "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
            "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
            "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
            "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
            "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
            "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
            "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
            "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
            "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
            "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
            "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
            "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
            "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
            "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
            "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
            "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
        }

        self._codon2aa_char: Dict[int, str] = {}
        self._aa2codons_char: Dict[str, List[int]] = {ch: [] for ch in "ACDEFGHIKLMNPQRSTVWY*"}
        for codon in self.codons:
            cid = self.vocab[codon]
            aa = self._genetic_code.get(codon, "X")
            self._codon2aa_char[cid] = aa
            if aa in self._aa2codons_char:
                self._aa2codons_char[aa].append(cid)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_ids(self) -> SpecialIds:
        return self._special_ids

    @property
    def num_special_tokens(self) -> int:
        return self._num_special_tokens

    @property
    def pad_token_id(self) -> int:
        return self._special_ids.pad

    @property
    def eos_token_id(self) -> int:
        return self._special_ids.eos

    # helpers
    def codon_vocab(self) -> Dict[str, int]:
        return {c: self.vocab[c] for c in self.codons}

    def codon2aa_char_map(self) -> Dict[int, str]:
        return dict(self._codon2aa_char)

    def aa2codons_char_map(self) -> Dict[str, List[int]]:
        return {k: v[:] for k, v in self._aa2codons_char.items()}

    # decoding
    def decode_codon_seq(self, token_ids: List[int]) -> str:
        parts: List[str] = []
        nst = self._num_special_tokens
        for tid in token_ids:
            if tid >= nst:
                tok = self.ids_to_tokens.get(tid)
                if tok is not None:
                    parts.append(tok)
        return "".join(parts)

    # persistence
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        payload = {
            "vocab": self.vocab,
            "special_token_str": self._special_token_str,
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "CodonTokenizer":
        vocab_path = Path(pretrained_model_name_or_path) / "vocab.json"
        tok = cls(**kwargs)
        if not vocab_path.exists():
            return tok
        with open(vocab_path, "r", encoding="utf-8") as f:
            save_data = json.load(f)
        vocab = save_data["vocab"] if isinstance(save_data, dict) and "vocab" in save_data else save_data
        tok.vocab = {str(k): int(v) for k, v in vocab.items()}
        tok.ids_to_tokens = {int(v): str(k) for k, v in tok.vocab.items()}
        sts = save_data.get("special_token_str", tok._special_token_str) if isinstance(save_data, dict) else tok._special_token_str
        tok._special_token_str.update(sts)
        def _id_for(name: str, default_val: int) -> int:
            sym = tok._special_token_str[name]
            return int(tok.vocab.get(sym, default_val))
        tok._special_ids = SpecialIds(
            pad=_id_for("pad", 0),
            unk=_id_for("unk", 1),
            bos=_id_for("bos", 2),
            eos=_id_for("eos", 3),
        )
        ids = [tok._special_ids.pad, tok._special_ids.unk, tok._special_ids.bos, tok._special_ids.eos]
        m = max(ids)
        tok._num_special_tokens = m + 1 if ids == list(range(m + 1)) else 4
        # rebuild helpers
        tok._codon2aa_char = {}
        tok._aa2codons_char = {ch: [] for ch in "ACDEFGHIKLMNPQRSTVWY*"}
        for codon in tok.codons:
            cid = tok.vocab[codon]
            aa = tok._genetic_code.get(codon, "X")
            tok._codon2aa_char[cid] = aa
            if aa in tok._aa2codons_char:
                tok._aa2codons_char[aa].append(cid)
        return tok
