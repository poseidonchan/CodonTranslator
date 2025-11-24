from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from .layers import RMSNorm, TransformerBlock
from .tokenizer import SpecialIds


class FrozenESMCEncoder(nn.Module):
    """Optional ESM-C encoder; if esm isn't available, stays inactive."""
    def __init__(self, model_name: str = "esmc_300m", device: str = "cuda", dtype: str = "bf16"):
        super().__init__()
        self.model_name = model_name
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._autocast_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else None)
        try:
            from esm.models.esmc import ESMC  # type: ignore
            from esm.utils.constants.models import ESMC_300M, ESMC_600M  # type: ignore
        except Exception as e:
            raise ImportError(
                "ESM is required for CodonTranslator. Please install 'esm>=3.2.0'."
            ) from e
        if self.model_name == "esmc_300m":
            const = ESMC_300M; self.D_esm = 960
        elif self.model_name == "esmc_600m":
            const = ESMC_600M; self.D_esm = 1152
        else:
            raise ValueError(f"Unknown ESM model: {self.model_name}")
        self.model = ESMC.from_pretrained(model_name=const, device=self._device)
        self.tokenizer = self.model.tokenizer
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def tokenize(self, sequences: List[str], max_length: Optional[int] = None, add_special_tokens: bool = True, return_tensors: str = "pt"):
        if self.model is None:
            raise RuntimeError("ESM model not available")
        from esm.utils import encoding  # type: ignore
        from esm.utils.misc import stack_variable_length_tensors  # type: ignore
        pad = self.tokenizer.pad_token_id
        toks = []
        for s in sequences:
            t = encoding.tokenize_sequence(s, self.tokenizer, add_special_tokens=add_special_tokens)
            if max_length is not None and len(t) > max_length:
                t = t[:max_length]
            toks.append(t)
        input_ids = stack_variable_length_tensors(toks, constant_value=pad)
        attention_mask = (input_ids != pad)
        return input_ids, attention_mask

    @torch.no_grad()
    def encode_from_ids(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.BoolTensor] = None, return_dict: bool = True):
        if self.model is None:
            raise RuntimeError("ESM model not available")
        device = self.model.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
        if self._autocast_dtype is not None and device.type == "cuda":
            with torch.amp.autocast('cuda', dtype=self._autocast_dtype):
                outputs = self.model.forward(sequence_tokens=input_ids, sequence_id=attention_mask)
        else:
            outputs = self.model.forward(sequence_tokens=input_ids, sequence_id=attention_mask)
        return {"embeddings": outputs.embeddings, "attention_mask": attention_mask}

    def strip_special_tokens(self, embeddings: torch.FloatTensor, attention_mask: Optional[torch.BoolTensor] = None):
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 2
            lengths = lengths.clamp(min=1)
        else:
            B, L, D = embeddings.shape
            lengths = torch.full((B,), L - 2, device=embeddings.device)
        stripped = embeddings[:, 1:-1, :]
        return stripped, lengths


class TranslatorBackbone(nn.Module):
    def __init__(
        self,
        vocab_size: int = 79,
        hidden_size: int = 960,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        max_position_embeddings: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
        num_special_tokens: int = 13,
        special_ids: Optional[SpecialIds] = None,
        esm_model_name: str = "esmc_300m",
        esm_device: str = "cuda",
        esm_dtype: str = "bf16",
        max_protein_prefix: int = 0,
        max_species_prefix: int = 0,
        prepend_species: bool = True,
        prepend_protein: bool = True,
        species_embedding_dim: int = 1024,
        attn_impl: str = "gqa",
        num_kv_groups: int = 0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.special_ids = special_ids or SpecialIds()
        self.num_special_tokens = int(num_special_tokens)

        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_size)

        # Optional ESM protein encoder
        self.esm = None
        self.esm_ln = None
        if prepend_protein and esm_model_name:
            # Enforce ESM presence – raise if missing
            self.esm = FrozenESMCEncoder(esm_model_name, esm_device, esm_dtype)
            self.esm_ln = nn.Sequential(
                nn.Linear(self.esm.D_esm, self.hidden_size, bias=False),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            )
        self.species_embedding_dim = species_embedding_dim if prepend_species else 0
        self.species_ln = None
        if prepend_species:
            self.species_ln = nn.Sequential(
                nn.Linear(self.species_embedding_dim, self.hidden_size, bias=False),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_size),
            )

        self.max_protein_prefix = int(max_protein_prefix) if max_protein_prefix is not None else 0
        self.max_species_prefix = int(max_species_prefix) if max_species_prefix is not None else 0
        self.prepend_species = bool(prepend_species)
        self.prepend_protein = bool(prepend_protein) and (self.esm is not None)

        self.start_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        nn.init.normal_(self.start_embed, mean=0.0, std=0.02)

        self.attn_impl = str(attn_impl)
        kv_groups = int(num_kv_groups)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                num_kv_groups=(kv_groups if (kv_groups > 0 and self.attn_impl == "gqa") else None),
                qk_norm=False,
                attn_type=("mha" if self.attn_impl == "mha" else "gqa"),
            )
            for _ in range(self.num_layers)
        ])

        self.ln_f = RMSNorm(self.hidden_size, eps=layer_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.gradient_checkpointing = False

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        device = self.token_embed.weight.device
        return self.token_embed(token_ids.to(device))

    def build_prefix(
        self,
        batch_size: int,
        device: torch.device,
        species_tok_emb: Optional[torch.Tensor] = None,
        species_emb: Optional[torch.Tensor] = None,
        protein_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts: list[torch.Tensor] = []
        if self.prepend_species and self.species_ln is not None:
            if species_emb is not None:
                S = self.species_ln(species_emb.to(device=device, dtype=next(self.parameters()).dtype).unsqueeze(1))
                parts.append(S)
                parts.append(S)
            elif species_tok_emb is not None:
                S = species_tok_emb
                if getattr(self, "max_species_prefix", 0) > 0 and S.size(1) > self.max_species_prefix:
                    S = S[:, : self.max_species_prefix, :]
                S = self.species_ln(S.to(device=device, dtype=next(self.parameters()).dtype))
                parts.append(S)
                parts.append(S)

        if self.prepend_protein and self.esm is not None and protein_input is not None:
            prot_ids, prot_mask = protein_input
            esm_out = self.esm.encode_from_ids(prot_ids, prot_mask, return_dict=True)
            P, lengths = self.esm.strip_special_tokens(esm_out["embeddings"], prot_mask)
            if getattr(self, "max_protein_prefix", 0) > 0 and P.size(1) > self.max_protein_prefix:
                P = P[:, : self.max_protein_prefix, :]
                lengths = lengths.clamp(max=self.max_protein_prefix) if lengths is not None else None
            if P.size(1) > 0:
                P = self.esm_ln(P.to(device=device, dtype=next(self.parameters()).dtype))
                if lengths is not None:
                    Lp = P.size(1)
                    ar = torch.arange(Lp, device=device).unsqueeze(0)
                    valid = ar < lengths.unsqueeze(1)
                    P = P * valid.unsqueeze(-1)
                parts.append(P)

        if len(parts) == 0:
            empty = torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=next(self.parameters()).dtype)
            return empty, torch.zeros(batch_size, dtype=torch.long, device=device)

        prefix = torch.cat(parts, dim=1)
        with torch.no_grad():
            valid = (prefix.abs().sum(dim=-1) > 0)
            lengths = valid.sum(dim=1).to(torch.long)
        prefix_budget = max(0, int(self.max_position_embeddings) - 1)
        allow = torch.minimum(lengths, torch.tensor(prefix_budget, device=lengths.device, dtype=lengths.dtype))
        Lp_max = int(allow.max().item()) if allow.numel() > 0 else 0
        if prefix.size(1) > Lp_max:
            trimmed = prefix.new_zeros(prefix.size(0), Lp_max, prefix.size(2))
            for b in range(prefix.size(0)):
                lb = int(allow[b].item())
                if lb > 0:
                    trimmed[b, :lb, :] = prefix[b, :lb, :]
            prefix = trimmed
            lengths = allow
        else:
            lengths = allow
        return prefix, lengths

    def forward(self, codon_ids: torch.Tensor, cond: Dict[str, Any] = None, labels: Optional[torch.Tensor] = None, return_dict: bool = True, use_cache: bool = False, past_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, position_offset: int = 0) -> Dict[str, torch.Tensor]:
        batch_size, codon_len = codon_ids.shape
        device = codon_ids.device
        species_tok_emb = cond.get("species_tok_emb") if cond else None
        species_emb = cond.get("species_emb") if cond else None
        protein_input = cond.get("protein_input") if cond else None

        # Build prefix
        prefix, prefix_lengths = self.build_prefix(batch_size, device, species_tok_emb=species_tok_emb, species_emb=species_emb, protein_input=protein_input)
        start = self.start_embed.expand(batch_size, 1, -1)

        # KV cache path for incremental generation
        if past_kv is not None and codon_len > 0:
            x = self.embed_tokens(codon_ids)
            present_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for i, block in enumerate(self.blocks):
                kv_i = past_kv[i] if i < len(past_kv) else None
                out_blk = block(x, past_kv=kv_i, use_cache=True, position_offset=position_offset)
                x, kv_out = out_blk
                present_kv.append(kv_out)
            x = self.ln_f(x)
            logits_step = self.lm_head(x)
            return {"logits": logits_step[:, 0:0, :], "next_logits": logits_step[:, -1, :], "present_kv": present_kv, "prefix_len": prefix_lengths}

        # Non-incremental: build prefix+start+codon window
        codon_lens = torch.as_tensor([codon_len] * batch_size, device=device)
        capacity = max(0, int(self.max_position_embeddings))
        budget_after_prefix = torch.clamp(torch.as_tensor(capacity, device=device) - (prefix_lengths + 1), min=0)
        per_cap = torch.minimum(budget_after_prefix, codon_lens)
        max_cap = int(per_cap.max().item()) if per_cap.numel() > 0 else 0
        codon_emb = self.embed_tokens(codon_ids[:, :max_cap]) if max_cap > 0 else torch.zeros(batch_size, 0, self.hidden_size, device=device, dtype=start.dtype)
        seqs = []
        for b in range(batch_size):
            lp = int(prefix_lengths[b].item())
            cap = int(per_cap[b].item())
            parts = []
            if lp > 0:
                parts.append(prefix[b, :lp, :])
            parts.append(start[b, 0:1, :])
            if cap > 0:
                parts.append(codon_emb[b, :cap, :])
            seqs.append(torch.cat(parts, dim=0))
        x = rnn_utils.pad_sequence(seqs, batch_first=True)

        present_kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            blk_out = block(x, use_cache=use_cache, position_offset=0)
            if use_cache:
                x, kv = blk_out
                present_kv_list.append(kv)
            else:
                x = blk_out
        x = self.ln_f(x)
        logits_full = self.lm_head(x)

        next_logits_list = []
        if max_cap == 0:
            codon_logits = logits_full[:, 0:0, :]
            for b in range(batch_size):
                lp = int(prefix_lengths[b].item())
                pos_next = lp
                next_logits_list.append(logits_full[b, pos_next, :] if pos_next < logits_full.size(1) else logits_full[b, -1, :])
            next_logits = torch.stack(next_logits_list, dim=0)
        else:
            slices = []
            for b in range(batch_size):
                lp = int(prefix_lengths[b].item())
                cap = int(per_cap[b].item())
                sl = logits_full[b, lp : lp + cap, :] if cap > 0 else logits_full.new_zeros(0, self.vocab_size)
                slices.append(sl)
                pos_next = lp + cap
                next_logits_list.append(logits_full[b, pos_next, :] if pos_next < logits_full.size(1) else logits_full.new_zeros(self.vocab_size))
            codon_logits = rnn_utils.pad_sequence(slices, batch_first=True)
            next_logits = torch.stack(next_logits_list, dim=0)
        out = {"logits": codon_logits, "next_logits": next_logits, "prefix_len": prefix_lengths}
        if use_cache:
            out["present_kv"] = present_kv_list
        return out
