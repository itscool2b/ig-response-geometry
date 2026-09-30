#encode task language instructions to T5-XXL embeddings
#run once offline — T5-XXL is too large for VRAM during IG
#uses sentencepiece directly (avoids transformers tiktoken bug)
#
#Usage:
#  .venv/bin/python encode_task_lang.py                          # encode all tasks + baseline
#  .venv/bin/python encode_task_lang.py --task PickSingleYCB-v1  # encode one task only
#  .venv/bin/python encode_task_lang.py --task PickCube-v1 --shuffle-seed 42
#        # encode a token-order-permuted version for sanity C2 input randomization;
#        # writes to data/lang_embeds/{task}_shuffled_s{seed}.pt

import os
import argparse
import random
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download, HfApi
from experiment_io import file_hash
from pathlib import Path
from transformers import T5EncoderModel

#task instructions for ManiSkill3 tasks used in this project. The first three
#are the original Month 2 tasks encoded on the local 3080 Ti. PickSingleYCB-v1
#and TurnFaucet-v1 were added during Month 3 to satisfy the protocol's 4-task scope
#(PickSingle env ID match, TurnFaucet as a Panda-compatible OpenDrawer substitute;
#OpenCabinetDrawer-v1 is Fetch-only in this ManiSkill install and would require
#rewriting the action pipeline).
task2lang = {
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1": "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickSingleYCB-v1": "Pick up the object on the table and lift it to the target position.",
    "TurnFaucet-v1": "Turn the faucet handle to the target angle.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=None,
                   help="Encode this task only. Omit to encode every task in task2lang "
                        "plus the PAD/EOS baseline.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="If set, shuffle content SentencePiece token order (EOS remains last) with this seed before "
                        "T5 encoding. Output saved to {task}_shuffled_s{seed}.pt. Used by "
                        "sanity.py C2 input randomization.")
    p.add_argument("--output-dir", default="data/lang_embeds")
    p.add_argument("--revision", default=None, help="T5 repository revision, resolved to a commit and recorded")
    return p.parse_args()


def encode_one(task_id, instruction, sp, model, output_dir, shuffle_seed=None):
    print(f"encoding: {task_id} -> '{instruction}'"
          + (f" [shuffle_seed={shuffle_seed}]" if shuffle_seed is not None else ""))
    token_ids = sp.Encode(instruction)
    token_ids.append(1)  # T5 EOS token

    if shuffle_seed is not None:
        #Shuffle the SentencePiece token ids (not the embeddings) and re-encode
        #through T5. This changes the syntactic order but preserves the set of
        #tokens, which is what the C2 "input randomization" semantics
        #requires (attributions should not survive a destroyed word order).
        rng = random.Random(shuffle_seed)
        content_ids = token_ids[:-1]
        rng.shuffle(content_ids)
        token_ids = content_ids + [1]

    tokens = [sp.IdToPiece(i) for i in token_ids]
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(model.device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        embeds = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.detach()
    #pad to max_lang_cond_len=1024 for consistency with RDT
    pad_len = 1024 - embeds.shape[1]
    if pad_len > 0:
        embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len))
        attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_len))

    suffix = f"_shuffled_s{shuffle_seed}" if shuffle_seed is not None else ""
    save_path = os.path.join(output_dir, f"{task_id}{suffix}.pt")
    save_embedding(save_path, {"embeds": embeds.cpu(), "attn_mask": attn_mask.cpu(),
                   "tokens": tokens, "token_ids": token_ids, "task": task_id,
                   "instruction": instruction, "shuffle_seed": shuffle_seed,
                   "shuffle_mode": "pre_encoding_content_tokens_eos_fixed" if shuffle_seed is not None else "none",
                   "identity": model._tmlr_identity})
    print(f"  saved {save_path}: embeds={tuple(embeds.shape)}, mask={tuple(attn_mask.shape)}, "
          f"tokens={tokens}")


def encode_baseline(sp, model, output_dir):
    #T5 has no formal BOS token; we use [PAD, EOS] = [0, 1] as the minimal
    #valid sequence. This is the IG language baseline used by forward_fn_language.
    print("encoding baseline: BOS/EOS minimal sequence [PAD, EOS]")
    baseline_ids = [0, 1]
    baseline_tokens = [sp.IdToPiece(i) for i in baseline_ids]
    input_ids = torch.tensor([baseline_ids], dtype=torch.long).to(model.device)
    attn_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        embeds = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state.detach()
    pad_len = 1024 - embeds.shape[1]
    if pad_len > 0:
        embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len))
        attn_mask = torch.nn.functional.pad(attn_mask, (0, pad_len))
    baseline_path = os.path.join(output_dir, "baseline_bos_eos.pt")
    save_embedding(baseline_path, {"embeds": embeds.cpu(), "attn_mask": attn_mask.cpu(),
                "tokens": baseline_tokens, "token_ids": baseline_ids,
                "baseline": "encoded_PAD_EOS_zero_padded", "identity": model._tmlr_identity})
    print(f"  saved {baseline_path}: embeds={tuple(embeds.shape)}, mask={tuple(attn_mask.shape)}, "
          f"tokens={baseline_tokens}")


def save_embedding(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Embedding exists; use a new output directory: {path}")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    args = parse_args()

    if args.task is not None and args.task not in task2lang:
        print(f"ERROR: unknown task {args.task}. Known tasks: {list(task2lang)}")
        raise SystemExit(1)

    #load sentencepiece tokenizer directly (bypasses transformers conversion bug)
    model_name = "google/t5-v1_1-xxl"
    print(f"loading sentencepiece from {model_name}...")
    revision = HfApi().model_info(model_name, revision=args.revision).sha
    spiece_path = hf_hub_download(model_name, "spiece.model", revision=revision)
    sp = spm.SentencePieceProcessor()
    sp.Load(spiece_path)

    #load T5 encoder with auto device mapping
    print(f"loading T5 encoder with device_map='auto'...")
    model = T5EncoderModel.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.train(False)
    model._tmlr_identity = {"model_repository": model_name, "model_revision": revision,
        "tokenizer_sha256": file_hash(spiece_path), "encoding_dtype": "torch.bfloat16",
        "source_sha256": file_hash(__file__), "historical_identity": "unknown; newly generated"}

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.task is not None:
        encode_one(args.task, task2lang[args.task], sp, model, output_dir,
                   shuffle_seed=args.shuffle_seed)
        if not Path(output_dir, "baseline_bos_eos.pt").exists():
            encode_baseline(sp, model, output_dir)
    else:
        if args.shuffle_seed is not None:
            print("ERROR: --shuffle-seed requires --task.")
            raise SystemExit(1)
        for task_id, instruction in task2lang.items():
            encode_one(task_id, instruction, sp, model, output_dir)
        encode_baseline(sp, model, output_dir)

    print("done — language embeddings saved")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
