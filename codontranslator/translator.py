from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file

from .models import TranslatorBackbone
from .tokenizer import CodonTokenizer
 # no external store at inference; species embeddings computed via Qwen


class CodonTranslator:
    """
    High-level sampling wrapper for trained checkpoints with a simple API:

        from CodonTranslator import CodonTranslator
        model = CodonTranslator.from_pretrained(model_path)
        dna = model.sampling(species="Homo sapiens", protein_seq="M...", enforce_mapping=True)
    """

    def __init__(self, model_dir: Union[str, Path], device: str = "cuda", use_gbif: bool = False):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.tokenizer = CodonTokenizer.from_pretrained(str(self.model_dir))
        self.V = int(self.tokenizer.vocab_size)
        self._eos_id = int(self.tokenizer.eos_token_id)
        self._pad_id = int(self.tokenizer.pad_token_id)
        self._num_special = int(self.tokenizer.num_special_tokens)

        # Load config
        cfg_path = self.model_dir / "trainer_config.json"
        if not cfg_path.exists():
            cfg_path = self.model_dir / "config.json"
        with open(cfg_path, "r") as f:
            self.config = json.load(f)

        # Build model and load weights
        state = self._load_state_dict()
        arch = self._infer_arch_from_state_dict(state)
        self.model = TranslatorBackbone(
            vocab_size=self.V,
            hidden_size=int(arch["hidden_size"]),
            num_layers=int(arch["num_layers"]),
            num_heads=int(arch["num_heads"]),
            mlp_ratio=float(arch.get("mlp_ratio", 4.0)),
            max_position_embeddings=int(arch["max_position_embeddings"]),
            num_special_tokens=self._num_special,
            special_ids=self.tokenizer.special_ids,
            prepend_species=bool(arch.get("prepend_species", True)),
            prepend_protein=bool(arch.get("prepend_protein", False)),
            species_embedding_dim=int(self.config.get("species_embedding_dim", 1024)),
            esm_model_name=str(arch.get("esm_model_name", "esmc_300m")),
            esm_device=str(arch.get("esm_device", "cuda")),
            esm_dtype=str(arch.get("esm_dtype", "bf16")),
            max_protein_prefix=int(arch.get("max_protein_prefix", 0)),
            max_species_prefix=int(arch.get("max_species_prefix", 0)),
            attn_impl=str(arch.get("attn_impl", "gqa")),
            num_kv_groups=int(arch.get("num_kv_groups", 0)),
        )
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            # non-fatal
            pass
        self.model.to(self.device).eval()

        # Static masks
        self._allowed_fixed = torch.ones(self.V, dtype=torch.bool, device=self.device)
        self._allowed_fixed[:self._num_special] = False
        self._allowed_variable = torch.ones(self.V, dtype=torch.bool, device=self.device)
        self._allowed_variable[:self._num_special] = False
        self._allowed_variable[self._eos_id] = True

        # Species taxonomy: either query GBIF (if allowed) or use raw names.
        self._use_gbif = bool(use_gbif)
        self._taxonomy_cache: Dict[str, str] = {}

    # ---- constructors ----
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path], device: str = "cuda", use_gbif: bool = False) -> "CodonTranslator":
        return cls(model_path, device=device, use_gbif=use_gbif)

    # ---- sampling APIs ----
    @torch.no_grad()
    def sampling(self, species: str, protein_seq: str, enforce_mapping: bool = False, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None, seed: Optional[int] = None, use_kv_cache: bool = True) -> str:
        out = self.batch_inference(
            species=[species],
            protein_seqs=[protein_seq],
            enforce_mapping=enforce_mapping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            use_kv_cache=use_kv_cache,
        )
        return out[0]

    @torch.no_grad()
    def batch_inference(
        self,
        species: List[str],
        protein_seqs: List[str],
        enforce_mapping: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        use_kv_cache: bool = True,
        micro_batch_size: int = 1,
    ) -> List[str]:
        """Generate DNA for a list of protein sequences, using micro-batching to limit memory.

        - micro_batch_size: number of samples to process at once (default=1 for low memory)
        """
        assert len(species) == len(protein_seqs), "species and protein_seqs length must match"
        mb = max(1, int(micro_batch_size))
        if len(species) <= mb:
            return self._batch_inference_core(
                species=species,
                protein_seqs=protein_seqs,
                enforce_mapping=enforce_mapping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                use_kv_cache=use_kv_cache,
            )

        outputs: List[str] = []
        for start in range(0, len(species), mb):
            end = min(start + mb, len(species))
            chunk_out = self._batch_inference_core(
                species=species[start:end],
                protein_seqs=protein_seqs[start:end],
                enforce_mapping=enforce_mapping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                use_kv_cache=use_kv_cache,
            )
            outputs.extend(chunk_out)
        return outputs

    @torch.no_grad()
    def _batch_inference_core(
        self,
        species: List[str],
        protein_seqs: List[str],
        enforce_mapping: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        use_kv_cache: bool = True,
    ) -> List[str]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        B = len(species)
        assert B == len(protein_seqs), "species and protein_seqs length must match"
        target_lens = torch.tensor([len(s) for s in protein_seqs], device=self.device, dtype=torch.long)
        T_codons = int(target_lens.max().item())

        # Prepare conditioning
        cond: Dict[str, Any] = {"control_mode": "fixed"}

        # Species embeddings via Qwen3-Embedding (variable-length token sequences)
        q_tok, lengths = self._qwen_embed_names(species, pooling="sequence")  # [B, L, D]
        # Always surface a message so users can see species embeddings are used
        print(f"[CodonTranslator] Species embeddings (Qwen) computed: shape={tuple(q_tok.shape)}")
        cond["species_tok_emb"] = q_tok.to(self.device)

        # Protein input via ESM (if available) – let model tokenize internally
        if getattr(self.model, "esm", None) is not None:
            # Tokenize AA sequences with model.esm
            max_len_tokens = (getattr(self.model, "max_protein_prefix", 0) + 2) if getattr(self.model, "max_protein_prefix", 0) > 0 else None
            prot_ids, prot_mask = self.model.esm.tokenize(protein_seqs, max_length=max_len_tokens)
            cond["protein_input"] = (prot_ids.to(self.device), prot_mask.to(self.device))

        # Start generation with empty context to build KV cache and initial logits
        input_ids = torch.zeros(B, 0, dtype=torch.long, device=self.device)
        out_prefill = self.model(codon_ids=input_ids, cond=cond, return_dict=True, use_cache=use_kv_cache)
        kv = out_prefill.get("present_kv") if use_kv_cache else None
        logits = out_prefill.get("next_logits")
        assert logits is not None
        # Report prefix length to prove species/protein prefixes were incorporated
        try:
            pref = out_prefill.get("prefix_len")
            if pref is not None:
                lst = pref.detach().cpu().tolist()
                print(f"[CodonTranslator] Prefix lengths (species,species,protein): {lst}")
        except Exception:
            pass

        allowed = self._allowed_fixed
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)

        aa2codons = self.tokenizer.aa2codons_char_map()

        rng = range(T_codons)
        # Greedy mode: temperature <= 0 selects argmax deterministically
        greedy_mode = (temperature is not None and float(temperature) <= 0.0)
        for step in rng:
            logits = logits.masked_fill(~allowed, float("-inf"))

            # Stop sampling per-sample once reaching its target length; force PAD
            done_now = (torch.tensor(step, device=self.device) >= target_lens)
            if done_now.any():
                logits[done_now] = float("-inf")
                logits[done_now, self._pad_id] = 0.0

            # Enforce codon ↔ AA mapping at this step
            if enforce_mapping:
                aas_now = [seq[step] if step < len(seq) else None for seq in protein_seqs]
                mask = torch.zeros_like(logits, dtype=torch.bool)
                for i, a in enumerate(aas_now):
                    if a is None:
                        mask[i, self._num_special:self.V] = True
                    else:
                        valid = aa2codons.get(a, [])
                        if len(valid) == 0:
                            mask[i, self._num_special:self.V] = True
                        else:
                            mask[i, valid] = True
                logits = logits.masked_fill(~mask, float("-inf"))

            if not greedy_mode and temperature != 1.0:
                logits = logits / float(temperature)
            if top_k is not None:
                logits = self._top_k_filtering(logits, int(top_k))
            if top_p is not None:
                logits = self._top_p_filtering(logits, float(top_p))

            if greedy_mode:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_tok], dim=1)

            if use_kv_cache:
                pos_offset = int(out_prefill.get("prefix_len").max().item()) + input_ids.size(1) - 1 if isinstance(out_prefill, dict) and ("prefix_len" in out_prefill) else input_ids.size(1) - 1
                out_inc = self.model(
                    codon_ids=next_tok,
                    cond=None,
                    return_dict=True,
                    use_cache=True,
                    past_kv=kv,
                    position_offset=pos_offset,
                )
                kv = out_inc.get("present_kv")
                logits = out_inc.get("next_logits")
            else:
                # Recompute full forward with prefix+all generated tokens
                out_full = self.model(codon_ids=input_ids, cond=cond, return_dict=True, use_cache=False)
                logits = out_full.get("next_logits")

        # Build DNA strings, dropping specials
        output_token_rows: List[List[int]] = []
        for i, row in enumerate(input_ids.tolist()):
            toks: List[int] = []
            for t in row:
                if t == self._pad_id:
                    continue
                if t == self._eos_id:
                    break
                if t >= self._num_special and t < self.V:
                    toks.append(int(t))
            toks = toks[: int(target_lens[i].item())]
            output_token_rows.append(toks)
        sequences = [self.tokenizer.decode_codon_seq(row) for row in output_token_rows]

        # If not enforcing mapping, report AA token accuracy vs provided targets
        if not enforce_mapping:
            for i, dna in enumerate(sequences):
                tgt = protein_seqs[i]
                gen_aa = self._dna_to_aa(dna)
                L = min(len(gen_aa), len(tgt))
                if L == 0:
                    acc = 0.0; num = 0; den = 0
                else:
                    num = sum(1 for a, b in zip(gen_aa[:L], tgt[:L]) if a == b)
                    den = L
                    acc = num / den
                print(f"[CodonTranslator] AA token accuracy seq_{i+1}: {acc:.4f} ({num}/{den})")
        return sequences

    # ---- helpers ----
    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        st_p = self.model_dir / "model.safetensors"
        if st_p.exists():
            return load_file(st_p)
        pt_p = self.model_dir / "pytorch_model.bin"
        if pt_p.exists():
            return torch.load(pt_p, map_location="cpu")
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {self.model_dir}")

    def _infer_arch_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        arch: Dict[str, Any] = {}
        if "lm_head.weight" in state_dict:
            arch["hidden_size"] = int(state_dict["lm_head.weight"].shape[1])
        else:
            for k, v in state_dict.items():
                if k.endswith("ln_f.weight"):
                    arch["hidden_size"] = int(v.shape[0])
                    break
        cfg = self.config or {}
        if "hidden_size" in cfg:
            arch["hidden_size"] = int(cfg["hidden_size"])  # type: ignore
        if "hidden_size" not in arch:
            arch["hidden_size"] = int(cfg.get("hidden_size", 750))
        H = int(arch["hidden_size"])

        max_block = -1
        for k in state_dict.keys():
            if k.startswith("blocks."):
                idx = int(k.split(".")[1])
                if idx > max_block:
                    max_block = idx
        arch["num_layers"] = (max_block + 1) if max_block >= 0 else int(cfg.get("num_hidden_layers", 12))
        if "num_hidden_layers" in cfg:
            arch["num_layers"] = int(cfg["num_hidden_layers"])  # type: ignore

        # mlp ratio
        w1_key = next((k for k in state_dict.keys() if k.endswith("ffn.w1.weight")), None)
        if w1_key is not None:
            arch["mlp_ratio"] = float(int(state_dict[w1_key].shape[0]) / H)
        else:
            arch["mlp_ratio"] = float(cfg.get("mlp_ratio", 4.0))

        # heads: pick divisor
        cfg_heads = cfg.get("num_attention_heads")
        if isinstance(cfg_heads, int) and cfg_heads > 0 and H % cfg_heads == 0:
            arch["num_heads"] = int(cfg_heads)
        else:
            for h in (16, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1):
                if H % h == 0:
                    arch["num_heads"] = h
                    break

        arch["prepend_species"] = bool(cfg.get("prepend_species", any(k.startswith("species_ln.") for k in state_dict.keys())))
        has_esm = any(k.startswith("esm_ln.") for k in state_dict.keys()) or any(k.startswith("esm.") for k in state_dict.keys())
        arch["prepend_protein"] = bool(cfg.get("prepend_protein", bool(has_esm)))
        arch["esm_model_name"] = str(cfg.get("esm_model_name", "esmc_300m"))
        arch["esm_device"] = str(cfg.get("esm_device", "cuda"))
        arch["esm_dtype"] = str(cfg.get("esm_dtype", "bf16")).lower()
        arch["max_protein_prefix"] = int(cfg.get("max_protein_prefix", 0))
        arch["max_species_prefix"] = int(cfg.get("max_species_prefix", 0))
        arch["max_position_embeddings"] = int(cfg.get("max_length", cfg.get("max_position_embeddings", 2048)))
        arch["attn_impl"] = str(cfg.get("attn_impl", "gqa"))
        arch["num_kv_groups"] = int(cfg.get("num_kv_groups", 0))
        return arch

    # --- filtering helpers
    @staticmethod
    def _ensure_2d_logits(logits: torch.Tensor) -> torch.Tensor:
        return logits if logits.dim() == 2 else logits.unsqueeze(0)

    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
        x = CodonTranslator._ensure_2d_logits(logits)
        k = max(1, min(int(k), x.size(-1)))
        values, _ = torch.topk(x, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        x = torch.where(x < min_values, torch.full_like(x, float('-inf')), x)
        return x if logits.dim() == 2 else x.squeeze(0)

    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
        if p >= 1.0:
            return logits
        if p <= 0.0:
            return torch.full_like(logits, float('-inf'))
        x = CodonTranslator._ensure_2d_logits(logits)
        sorted_logits, sorted_indices = torch.sort(x, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        to_remove = cumprobs > p
        # Avoid overlapping memory writes by cloning the RHS
        to_remove = to_remove.to(torch.bool)
        to_remove[:, 1:] = to_remove[:, :-1].clone()
        to_remove[:, 0] = False
        mask = torch.zeros_like(x, dtype=torch.bool).scatter(-1, sorted_indices, to_remove)
        x = torch.where(mask, torch.full_like(x, float('-inf')), x)
        return x if logits.dim() == 2 else x.squeeze(0)

    # --- Qwen embedding fallback for species text ---
    def _qwen_embed_names(self, names: List[str], pooling: str = "sequence") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True, padding_side="left"
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", dtype=dtype, trust_remote_code=True
        ).to(self.device).eval()
        task = (
            "Given a species taxonomy information, generate a biological embedding "
            "representing its taxonomic and evolutionary characteristics"
        )
        queries = self._resolve_taxonomy_texts(names)
        texts = [f"Instruct: {task}\nQuery: {q}" for q in queries]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        out = model(**inputs)
        h = torch.nn.functional.normalize(out.last_hidden_state, p=2, dim=-1)
        attn = inputs["attention_mask"]
        # sequence embeddings padded to same length by tokenizer padding
        return h, torch.sum(attn, dim=1)

    def _taxonomy_lookup(self, name: str) -> str:
        if name in self._taxonomy_cache:
            return self._taxonomy_cache[name]
        if self._use_gbif:
            try:
                import requests
                resp = requests.get("https://api.gbif.org/v1/species/match", params={"name": name}, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("matchType") != "NONE":
                        parts = []
                        taxonomy = []
                        for rank in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]:
                            if rank in data and data[rank]:
                                taxonomy.append(data[rank])
                        if taxonomy:
                            parts.append("Taxonomy: " + " > ".join(taxonomy))
                        if "vernacularName" in data and data["vernacularName"]:
                            parts.append(f"Common name: {data['vernacularName']}")
                        if "confidence" in data:
                            parts.append(f"Match confidence: {data['confidence']}%")
                        if "status" in data:
                            parts.append(f"Status: {data['status']}")
                        desc = ". ".join(parts) if parts else name
                        self._taxonomy_cache[name] = desc
                        return desc
            except Exception:
                pass
        return name

    def _resolve_taxonomy_texts(self, names: List[str]) -> List[str]:
        """Resolve taxonomy strings for a batch of species names.
        If a taxonomy DB is present, pull from it. Otherwise batch-query GBIF
        (one request per species) and cache results. Always returns a list of
        strings aligned to `names`.
        """
        results: List[str] = []
        # Batch “query”: loop per-name; still batched at the embedding stage
        fetched = 0
        for s in names:
            txt = self._taxonomy_lookup(s)
            if s in self._taxonomy_cache:
                fetched += 1
            results.append(txt)
        if self._use_gbif:
            print(f"[CodonTranslator] Taxonomy texts resolved (GBIF={'on' if self._use_gbif else 'off'}): {fetched}/{len(names)} fetched")
        return results

    @staticmethod
    def _dna_to_aa(dna_seq: str) -> str:
        g = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        L = len(dna_seq) // 3
        aa = [g.get(dna_seq[3*i:3*i+3], 'X') for i in range(L)]
        return ''.join(aa)
