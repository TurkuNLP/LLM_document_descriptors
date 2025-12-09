#!/usr/bin/env python3
"""Collapse duplicate descriptor–explainer pairs by *exact descriptor match* using an LLM
that selects the most representative pair to keep within each duplicate group.

Inputs (JSONL, one object per line):
- id (str): unique id for the pair
- descriptor (str)
- explainer (str)
- source_pair_ids (list[str] | str | None): optional provenance ids; kept & aggregated on merges

Outputs (written into results/<run_id>/):
- <run_id>.jsonl                 : final kept pairs (id, descriptor, explainer, aggregated source ids)
- <run_id>_lineage.jsonl         : lineage edges for all merges
- <run_id>_settings.txt          : run settings & metadata
- <run_id>.groups.json           : mapping final kept_id -> list of member ids merged into it

Notes
-----
• Only *exact* duplicate descriptors are merged; groups of size 1 pass through unchanged.
• Within each duplicate group, an LLM receives all candidates and returns {"keep_id": "..."}.
• If the LLM output cannot be parsed, a deterministic tie‑breaker is used to avoid failures.
• No embeddings, no FAISS: this script is a single-pass by descriptor equality.
"""
from __future__ import annotations

# ===== Standard library =====
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import GuidedDecodingParams  # type: ignore

import json_repair  # type: ignore
import torch #type: ignore

from normalize_descriptor import normalize_descriptor

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Pair:
    id: str
    descriptor: str
    explainer: str
    source_ids: List[str]

    @property
    def descriptor_text(self) -> str:
        return (self.descriptor or "").strip()


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def setup_logging(logging_dir: Path, verbosity: int = 1) -> None:
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_file = logging_dir / f"{logging_dir.name}.log"
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8"); fh.setLevel(level); fh.setFormatter(fmt); root.addHandler(fh)
    sh = logging.StreamHandler(); sh.setLevel(level); sh.setFormatter(fmt); root.addHandler(sh)


def read_jsonl(path: Path, sample_size: Optional[int] = None) -> List[Pair]:
    pairs: List[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSON line %d", i)
                continue
            pid = str(obj.get("id", f"{i:08d}"))
            d = str(obj.get("descriptor", ""))
            e = str(obj.get("explainer", ""))
            src = obj.get("source_pair_ids")
            if src is None:
                source_ids: List[str] = []
            elif isinstance(src, list):
                source_ids = [str(s) for s in src]
            else:
                source_ids = [str(src)]
            if not d and not e:
                continue
            pairs.append(Pair(id=pid, descriptor=d, explainer=e, source_ids=source_ids))
            if sample_size is not None and len(pairs) >= sample_size:
                logging.info("Test mode: limiting to %d rows", sample_size)
                break
    logging.info("Loaded %d rows from %s", len(pairs), path)
    return pairs


def write_jsonl(path: Path, pairs: List[Pair]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps({
                "id": p.id,
                "descriptor": p.descriptor,
                "explainer": p.explainer,
                "source_pair_ids": p.source_ids,
            }, ensure_ascii=False) + "\n")
    logging.info("Wrote %d pairs -> %s", len(pairs), path)


# -----------------------------------------------------------------------------
# LLM wrapper
# -----------------------------------------------------------------------------

class GroupChooser:
    """Encapsulate the LLM that picks keep_id in a group."""
    def __init__(self, model_name: str, cache_dir: Optional[Path] = None, temperature: float = 0.1) -> None:
        self.model_name = model_name
        self.cache_dir = os.environ.get("HF_HOME") or (str(cache_dir) if cache_dir else None)
        self.temperature = temperature
        self._llm = None

    # ---- LLM bits -----------------------------------------------------------
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPU available.")
            logging.info("Loading LLM, using %d GPU(s).", n_gpus)
            self._llm = LLM(model=self.model_name,
                            download_dir=self.cache_dir, 
                            dtype="bfloat16", 
                            max_model_len=2048, 
                            tensor_parallel_size=n_gpus, 
                            enforce_eager=False, 
                            gpu_memory_utilization=0.85)
        return self._llm

    @staticmethod
    def _schema() -> GuidedDecodingParams:
        # minimal JSON schema: {"keep_id": string}
        return GuidedDecodingParams(json={
            "type": "object",
            "properties": {"keep_id": {"type": "string"}},
            "required": ["keep_id"],
            "additionalProperties": False,
        })

    @staticmethod
    def _prompt(cands: List[Pair]) -> str:
        lines = []
        for p in cands:
            lines.append(f"id: {p.id} | descriptor: {p.descriptor} | explainer: {p.explainer}")
        joined = "\n".join(lines)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            "You are selecting a representative pair among candidates that share the same descriptor.\n"
            "The descriptor describes a document, while the explainer provides additional context.\n"
            "Your task is to choose the most representative pair among the given pairs.\n"
            "When choosing the representative pair, prefer concise, general, commonly used phrasing that covers the meaning of all pairs.\n"
            "Return ONLY JSON with: keep_id (string). Do not invent or rewrite anything.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" + joined +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )

    def choose_keep_id(self, cands: List[Pair]) -> str:
        if len(cands) == 1:
            return cands[0].id
        _ = self.llm
        assert self._llm is not None
        prompts = [self._prompt(cands)]
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=64,
            top_p=0.5,
            stop=["<|eot_id|>"],
            guided_decoding=self._schema(),
        )
        out = self._llm.generate(prompts, params)[0]
        text = (out.outputs[0].text if out.outputs else "").strip()
        try:
            if json_repair is not None:
                obj = json_repair.loads(text)
            else:
                obj = json.loads(text)
            keep = str(obj.get("keep_id"))
            if keep and any(p.id == keep for p in cands):
                return keep
        except Exception:
            logging.warning("LLM parse failure; falling back to deterministic tie-breaker. Raw: %r", text)
        # Deterministic tie-breaker on failure: prefer non-empty explainer, then shorter descriptor, then lexicographic id
        def key(p: Pair):
            return (0 if p.explainer.strip() else 1, len(p.descriptor.strip()), p.id)
        return sorted(cands, key=key)[0].id


# -----------------------------------------------------------------------------
# Core merge logic
# -----------------------------------------------------------------------------

def group_by_descriptor(pairs: List[Pair]) -> Dict[str, List[Pair]]:
    groups: Dict[str, List[Pair]] = {}
    for p in pairs:
        key = normalize_descriptor(p.descriptor_text)
        if not key and not (p.explainer or "").strip():
            continue
        groups.setdefault(key, []).append(p)
    return groups


def reduce_groups(groups: Dict[str, List[Pair]], chooser: GroupChooser) -> Tuple[List[Pair], List[dict], Dict[str, List[str]]]:
    """Return (final_pairs, lineage_records, groups_mapping)."""
    final_pairs: List[Pair] = []
    lineage: List[dict] = []
    kept_to_members: Dict[str, List[str]] = {}

    for key, members in groups.items():
        if len(members) == 1:
            final_pairs.append(members[0])
            kept_to_members[members[0].id] = [members[0].id]
            continue
        keep_id = chooser.choose_keep_id(members)
        kept = next(p for p in members if p.id == keep_id)
        # Aggregate provenance from all members into kept
        all_src: List[str] = list(kept.source_ids)
        for m in members:
            if m is kept:
                continue
            lineage.append({
                "event_type": "synonym_merge",
                "descriptor": kept.descriptor,
                "new_pair_id": kept.id,
                "source_pair_ids": [m.id],
                "kept": {"id": kept.id, "descriptor": kept.descriptor, "explainer": kept.explainer},
                "dropped": {"id": m.id, "descriptor": m.descriptor, "explainer": m.explainer},
                "decision_reason": "LLM_decision",
            })
            all_src.extend(m.source_ids if m.source_ids else [])
        kept_agg = Pair(id=kept.id, descriptor=kept.descriptor, explainer=kept.explainer, source_ids=sorted(set(all_src)))
        final_pairs.append(kept_agg)
        kept_to_members[kept.id] = [m.id for m in members]

    # Stable sort final_pairs by id for determinism
    final_pairs.sort(key=lambda p: p.id)
    return final_pairs, lineage, kept_to_members

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Collapse duplicate descriptors via LLM representative selection")
    ap.add_argument("--run-id", required=True, help="Run identifier (used for output folder & file names)")
    ap.add_argument("--input", required=True, help="Path to input JSONL with fields: id, descriptor, explainer, source_pair_ids")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--verbose", type=int, default=1, help="0=warn, 1=info, 2=debug")
    ap.add_argument("--test", type=int, default=None, help="Limit number of input rows (for quick tests)")
    args = ap.parse_args()

    out_dir = Path("../results") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir, args.verbose)
    with (out_dir / f"{args.run_id}_settings.txt").open("w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            logging.info("%s: %s", k, v); f.write(f"{k}: {v}\n")

    pairs = read_jsonl(Path(args.input), sample_size=args.test)

    chooser = GroupChooser(model_name=args.model, temperature=args.temperature)
    groups = group_by_descriptor(pairs)
    final_pairs, lineage, kept_to_members = reduce_groups(groups, chooser)

    # Write outputs
    write_jsonl(out_dir / f"{args.run_id}.jsonl", final_pairs)

    with (out_dir / f"{args.run_id}_lineage.jsonl").open("w", encoding="utf-8") as f:
        for rec in lineage:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote lineage -> %s", out_dir / f"{args.run_id}_lineage.jsonl")

    with (out_dir / f"{args.run_id}.groups.json").open("w", encoding="utf-8") as f:
        json.dump(kept_to_members, f, ensure_ascii=False, indent=2)
    logging.info("Wrote groups -> %s", out_dir / f"{args.run_id}.groups.json")


if __name__ == "__main__":
    main()