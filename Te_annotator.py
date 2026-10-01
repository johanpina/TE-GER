import typer
import torch
import warnings
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, AutoConfig
from Bio import SeqIO
from tqdm import tqdm
from typing import List, Dict
import torch.nn as nn
import json
import time

# Filter warnings
warnings.filterwarnings("ignore")

app = typer.Typer(
    name="DNABERT-2 TE Annotator",
    help="CLI for TE detection at Binary, Order, or Superfamily levels."
)

# --- PATH CONFIGURATION ---
BASE_MODELS_PATH = "./models"

# --- HuggingFace repos for each classification level ---
HF_REPOS = {
    "binary":       "Jspinad/te-ger-binary",
    "order":        "Jspinad/te-ger-order",
    "superfamilies": "Jspinad/te-ger-superfamilies",
}

def ensure_model_weights(level: str, model_dir: str) -> None:
    """Download model weights from HuggingFace if not present locally."""
    required = ["config.json", "pytorch_model.bin"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_dir, f))]

    if not missing:
        return

    repo_id = HF_REPOS.get(level)
    if repo_id is None:
        typer.echo(f"❌ No HuggingFace repo configured for level '{level}'.")
        raise typer.Exit(1)

    typer.echo(f"📥 Model weights not found locally. Downloading from HuggingFace: {repo_id}")
    typer.echo(f"   Destination: {model_dir}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        typer.echo("❌ 'huggingface_hub' is not installed. Run: pip install huggingface_hub")
        raise typer.Exit(1)

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=model_dir,
        ignore_patterns=["*.gitkeep", ".gitattributes"],
    )
    typer.echo(f"✅ Weights downloaded successfully.")

os.environ["TOKENIZERS_PARALLELISM"] = 'True'

# --- 1. HYBRID ARCHITECTURE DEFINITION ---
class DNABERT_BiLSTM_NER(nn.Module):
    def __init__(self, checkpoint, num_labels, id2label, label2id):
        super().__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.config = self.bert.config
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id

        input_dim = self.config.hidden_size
        lstm_hidden_dim = 256
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Flatten for multi-GPU optimization
        self.bilstm.flatten_parameters()

        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(self.dropout(lstm_output))
        return logits

# --- 2. DATASET (MEGA-CHUNKS) ---
class GenomeChunkDataset(Dataset):
    # NOTE: Reduced from 1MB to 200kb.
    # Perfect balance: saturates all GPUs without risk of OOM.
    def __init__(self, fasta_path: str, tokenizer, chunk_size=200000, window_size=512, stride=128):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.chunk_size = chunk_size
        self.chunks_metadata = []

        print(f"📖 Indexing FASTA sequence: {fasta_path}...")
        begin = time.time()
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_len = len(record.seq)
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                self.chunks_metadata.append({
                    "seq_id": record.id,
                    "seq_str": str(record.seq[start:end]).upper(),
                    "global_start": start
                })
        print(f"LOG: Sequence indexed in {(time.time() - begin):.2f}s. Total Chunks: {len(self.chunks_metadata)}")

    def __len__(self):
        return len(self.chunks_metadata)

    # ... (__getitem__ unchanged) ...
    def __getitem__(self, idx):
        meta = self.chunks_metadata[idx]
        tokens = self.tokenizer(
            meta["seq_str"],
            truncation=True,
            max_length=self.window_size,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "offset_mapping": tokens["offset_mapping"],
            "global_start": meta["global_start"],
            "seq_id": meta["seq_id"]
        }

# --- 3. POST-PROCESSING ---
def _start_conf_acc(d):
    """Inicializa el acumulador de confianza ponderada por longitud (in-place)."""
    w = max(d['end'] - d['start'], 1)
    d['_wsum'] = d.get('conf', 0.0) * w
    d['_wlen'] = w

def _finalize_conf(d):
    """Cierra el acumulador: conf = media ponderada por longitud."""
    if '_wlen' in d:
        d['conf'] = d['_wsum'] / d['_wlen'] if d['_wlen'] > 0 else d.get('conf', 0.0)
        d.pop('_wsum', None); d.pop('_wlen', None)

def merge_annotations(raw_preds: List[Dict], gap_tolerance=10) -> List[Dict]:
    if not raw_preds: return []
    # Sorting is critical for merge
    raw_preds.sort(key=lambda x: (x['seq_id'], x['start']))
    merged = []
    # Copia para no mutar los dicts del llamador
    current = dict(raw_preds[0])
    has_conf = 'conf' in current
    if has_conf: _start_conf_acc(current)

    for next_pred in raw_preds[1:]:
        if (next_pred['seq_id'] == current['seq_id'] and
            next_pred['label'] == current['label'] and
            next_pred['start'] <= current['end'] + gap_tolerance):
            current['end'] = max(current['end'], next_pred['end'])
            if has_conf:
                w = max(next_pred['end'] - next_pred['start'], 1)
                current['_wsum'] += next_pred.get('conf', 0.0) * w
                current['_wlen'] += w
        else:
            if has_conf: _finalize_conf(current)
            merged.append(current)
            current = dict(next_pred)
            if has_conf: _start_conf_acc(current)
    if has_conf: _finalize_conf(current)
    merged.append(current)
    return merged

def filter_min_length(annotations: List[Dict], min_len: int) -> List[Dict]:
    """Descarta anotaciones más cortas que min_len pb. min_len<=0 -> sin filtro."""
    if min_len <= 0:
        return annotations
    return [a for a in annotations if (a['end'] - a['start']) >= min_len]

def write_gff3(annotations: List[Dict], output_path: str, source="TEGER", export_conf=False):
    print(f"💾 Saving {len(annotations)} annotations to {output_path}...")
    with open(output_path, "w") as f:
        f.write("##gff-version 3\n")
        for ann in annotations:
            # GFF is 1-based
            start = ann['start'] + 1
            end = ann['end']
            conf = ann.get('conf')
            score = f"{conf:.4f}" if (export_conf and conf is not None) else "."
            attrs = f"ID={ann['label']}_{start}_{end}"
            if export_conf and conf is not None:
                attrs += f";conf={conf:.4f}"
            line = f"{ann['seq_id']}\t{source}\t{ann['label']}\t{start}\t{end}\t{score}\t+\t.\t{attrs}\n"
            f.write(line)

def write_fasta_library(annotations: List[Dict], genome_path: str, output_path: str):
    """
    Generates a FASTA library from GFF annotations.
    """
    print(f"📚 Generating FASTA library at {output_path}...")
    try:
        genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))
        with open(output_path, "w") as f:
            for i, ann in enumerate(annotations):
                seq_id = ann['seq_id']
                start = ann['start'] # 0-based
                end = ann['end']
                label = ann['label']

                if seq_id in genome:
                    # Header format: >TE_1#LTR/Gypsy
                    header = f">TE_{i+1}#{label}"
                    sequence = str(genome[seq_id].seq[start:end])
                    f.write(f"{header}\n{sequence}\n")
    except Exception as e:
        print(f"⚠️ Error generating FASTA library: {e}")

# --- 4. MAIN CLI COMMAND ---
@app.command()
def predict(
    fasta_file: str = typer.Argument(..., help="Input FASTA file."),
    output_gff: str = typer.Argument(..., help="Output GFF3 annotation file."),
    level: str = typer.Option("binary", help="Classification level: binary, order, superfamilies."),
    create_library: bool = typer.Option(True, help="Generate a FASTA library of candidate TE sequences."),
    num_workers: int = typer.Option(4, help="CPU workers for data pre-processing."),
    chunk_size: int = typer.Option(1000000, help="Chunk size in base pairs. Adjust based on available VRAM."),
    device: str = typer.Option("cuda", help="Execution device (cuda/cpu)."),
    gpu_ids: str = typer.Option(None, help="Comma-separated GPU IDs to use (e.g., '0,1,2'). Defaults to all GPUs."),
    num_gpus: int = typer.Option(0, help="Maximum number of GPUs to use. 0 = use all available."),
    aggregate_windows: bool = typer.Option(
        False,
        help="Average softmax probabilities across overlapping windows per genomic "
             "span before argmax (resolves duplicate/conflicting labels at the same "
             "position). Default False = legacy per-window independent argmax."),
    te_threshold: float = typer.Option(
        None,
        help="Confidence gate: a span is called TE only if its max non-Background "
             "probability >= this value (0-1). Reduces over-prediction. "
             "Default None = legacy argmax (no threshold)."),
    min_len: int = typer.Option(
        0,
        help="Drop final merged annotations shorter than N bp. 0 = no filter (default)."),
    export_conf: bool = typer.Option(
        False,
        help="Write the mean confidence into GFF column 6 (score) and a ;conf= "
             "attribute. Default False = legacy '.' score.")
):
    # Preemptive GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    level = level.lower()
    model_dir = f"./models/{level}/"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # --- GPU RESOLUTION ---
    selected_device_ids = None  # None = use all (default DataParallel behavior)

    if device == "cuda" and torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()

        if gpu_ids is not None:
            # User specified exact IDs: --gpu-ids "0,2,3"
            try:
                selected_device_ids = [int(x.strip()) for x in gpu_ids.split(",")]
                # Validate that all IDs exist
                invalid_ids = [gid for gid in selected_device_ids if gid >= total_gpus or gid < 0]
                if invalid_ids:
                    typer.echo(f"⚠️ Invalid GPU IDs: {invalid_ids}. Available: 0-{total_gpus - 1}")
                    raise typer.Exit(1)
            except ValueError:
                typer.echo(f"❌ Invalid format for --gpu-ids. Use comma-separated integers, e.g.: '0,1,2'")
                raise typer.Exit(1)
        elif num_gpus > 0:
            # User specified number of GPUs: --num-gpus 4
            effective_num = min(num_gpus, total_gpus)
            selected_device_ids = list(range(effective_num))

        # GPU information
        if selected_device_ids is not None:
            print(f"🎯 Selected GPUs: {selected_device_ids} (of {total_gpus} available)")
            for gid in selected_device_ids:
                print(f"   GPU {gid}: {torch.cuda.get_device_name(gid)}")
            primary_device = f"cuda:{selected_device_ids[0]}"
        else:
            print(f"🎯 Using all available GPUs: {total_gpus}")
            for gid in range(total_gpus):
                print(f"   GPU {gid}: {torch.cuda.get_device_name(gid)}")
            primary_device = "cuda:0"

        device = primary_device

    print(f"Using device: {device} ⚙️")

    try:
        begin = time.time()

        # Ensure weights are present (downloads from HuggingFace on first run)
        if level in ["binary", "superfamilies", "binario", "superfamilia", "order", "orden"]:
            ensure_model_weights(level, model_dir)

        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config_data = json.load(f)

        id2label = {int(k): v for k, v in config_data["id2label"].items()}
        label2id = config_data["label2id"]
        num_labels = len(id2label)
        bg_id = label2id.get("Background", label2id.get("0", 0))

        if level in ["binary", "superfamilies", "binario", "superfamilia", "order", "orden"]:
            typer.echo(f"🧠 Loading Hybrid Model: {level}...")
            SAFE_CHECKPOINT = "quietflamingo/dnabert2-no-flashattention"
            model = DNABERT_BiLSTM_NER(SAFE_CHECKPOINT, num_labels, id2label, label2id)
            weights_path = os.path.join(model_dir, "pytorch_model.bin")

            state_dict = torch.load(weights_path, map_location=device)
            load_result = model.load_state_dict(state_dict, strict=False)

            # --- WEIGHT LOADING VERIFICATION ---
            total_keys = len(model.state_dict())
            loaded_keys = total_keys - len(load_result.missing_keys)
            print(f"📦 Weights loaded: {loaded_keys}/{total_keys} layers from {weights_path}")
            if load_result.missing_keys:
                print(f"   ⚠️ Layers NOT loaded (randomly initialized): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"   ⚠️ Layers in file but not in model: {load_result.unexpected_keys}")
            if loaded_keys == 0:
                typer.echo("❌ ALERT: No weights were loaded. Model is completely randomly initialized.")
                raise typer.Exit(1)
            elif len(load_result.missing_keys) > 0:
                print(f"   ℹ️ {len(load_result.missing_keys)} layers have random weights. Verify if expected.")
            else:
                print(f"   ✅ All weights loaded successfully.")
        else:
            typer.echo(f"🧬 Loading Standard Model: {level}...")
            model = AutoModelForTokenClassification.from_pretrained(model_dir, trust_remote_code=True)

        use_multi_gpu = (
            "cuda" in device
            and torch.cuda.device_count() > 1
            and (selected_device_ids is None or len(selected_device_ids) > 1)
        )
        if use_multi_gpu:
            ngpus = len(selected_device_ids) if selected_device_ids else torch.cuda.device_count()
            print(f"⚡ Multi-GPU Activated! Using {ngpus} GPUs with device_ids={selected_device_ids or 'all'}.")
            model = nn.DataParallel(model, device_ids=selected_device_ids)

        model.to(device).eval()
    except Exception as e:
        typer.echo(f"❌ Error loading model: {e}")
        raise typer.Exit(1)

    begin1 = time.time()

    # 200KB Chunk Size
    dataset = GenomeChunkDataset(
        fasta_path=fasta_file,
        tokenizer=tokenizer,
        chunk_size=chunk_size  # <--- USING THE VARIABLE
    )

    print(f"LOG: Dataset loaded. Chunk Size: {chunk_size/1000} kb.")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True if "cuda" in device else False
    )

    final_annotations = []

    # Activamos la ruta de probabilidad solo si se pide alguna mejora opt-in.
    # Sin flags -> prob_path=False -> rama legacy idéntica al comportamiento actual.
    prob_path = (te_threshold is not None) or export_conf or aggregate_windows
    if prob_path:
        print(f"🔬 Probability path ON (aggregate_windows={aggregate_windows}, "
              f"te_threshold={te_threshold}, export_conf={export_conf})")

    print("🚀 Starting Vectorized Inference (FP16)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):

            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            if input_ids.dim() == 3:
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)

            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # --- OPTIMIZATION: FP16 (Mixed Precision) ---
            # Halves VRAM usage and accelerates on RTX
            autocast_device = "cuda" if "cuda" in device else "cpu"
            with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                # Unflatten dimensions before leaving the GPU
                B_windows, L, _ = logits.shape
                logits = logits.view(B_windows, L, -1)

                if not prob_path:
                    # Argmax on GPU is faster
                    preds = torch.argmax(logits, dim=2).cpu().numpy()
                else:
                    # softmax en fp32 (NO fp16) para estabilidad del umbral
                    probs = torch.softmax(logits.float(), dim=2).cpu().numpy()  # (B, L, C)

            # --- VECTORIZED RECONSTRUCTION ---
            offset_mapping = batch['offset_mapping'][0].numpy()
            global_start = batch['global_start'].item()
            seq_id = batch['seq_id'][0]

            if not prob_path:
                # ---------- RAMA LEGACY (idéntica al comportamiento original) ----------
                valid_mask = (preds != bg_id) & (offset_mapping[:, :, 0] != offset_mapping[:, :, 1])

                if not np.any(valid_mask):
                    continue

                valid_labels_ids = preds[valid_mask]
                valid_starts_local = offset_mapping[:, :, 0][valid_mask]
                valid_ends_local = offset_mapping[:, :, 1][valid_mask]

                valid_starts_global = valid_starts_local + global_start
                valid_ends_global = valid_ends_local + global_start

                chunk_results = [
                    {
                        "seq_id": seq_id,
                        "start": int(s),
                        "end": int(e),
                        "label": id2label[l]
                    }
                    for s, e, l in zip(valid_starts_global, valid_ends_global, valid_labels_ids)
                ]
            else:
                # ---------- RAMA DE PROBABILIDAD (agregación + umbral + conf) ----------
                cs = offset_mapping[:, :, 0]
                ce = offset_mapping[:, :, 1]
                real = (cs != ce)  # descarta padding / tokens especiales (offset 0,0)

                if not np.any(real):
                    continue

                cs_r = cs[real]; ce_r = ce[real]
                probs_r = probs[real]  # (N_real, C)

                # Agrupa por span (char_start, char_end): el mismo span físico
                # comparte (cs,ce) entre ventanas solapadas -> una sola clave.
                keys = np.stack([cs_r, ce_r], axis=1)
                uniq, idx_first, inv = np.unique(
                    keys, axis=0, return_index=True, return_inverse=True)
                inv = inv.reshape(-1)  # defensivo: numpy 2.0 pudo devolver 2D
                C = probs_r.shape[1]
                if aggregate_windows:
                    # promedia softmax entre todas las ventanas que cubren el span
                    psum = np.zeros((len(uniq), C), dtype=np.float64)
                    np.add.at(psum, inv, probs_r)
                    counts = np.bincount(inv, minlength=len(uniq)).reshape(-1, 1)
                    mean_prob = psum / np.maximum(counts, 1)
                else:
                    # sin agregar: representante = primera ventana vista del span
                    mean_prob = probs_r[idx_first]

                # mejor clase TE (excluyendo Background) y su probabilidad
                te_prob = mean_prob.copy()
                te_prob[:, bg_id] = -1.0
                best = np.argmax(te_prob, axis=1)
                best_p = te_prob[np.arange(len(uniq)), best]
                bg_p = mean_prob[:, bg_id]

                # compuerta Background + umbral de confianza
                keep = best_p > bg_p
                if te_threshold is not None:
                    keep &= best_p >= te_threshold

                chunk_results = [
                    {
                        "seq_id": seq_id,
                        "start": int(uniq[k, 0]) + global_start,
                        "end": int(uniq[k, 1]) + global_start,
                        "label": id2label[int(best[k])],
                        "conf": float(best_p[k]),
                    }
                    for k in np.nonzero(keep)[0]
                ]

            if not chunk_results:
                continue

            chunk_merged = merge_annotations(chunk_results)
            final_annotations.extend(chunk_merged)

    print(f"LOG: Inference completed.")
    print("🧩 Performing final merge...")
    final_clean_annotations = merge_annotations(final_annotations)
    if min_len > 0:
        n_before = len(final_clean_annotations)
        final_clean_annotations = filter_min_length(final_clean_annotations, min_len)
        print(f"LOG: min_len={min_len} -> {n_before} -> {len(final_clean_annotations)} annotations")
    write_gff3(final_clean_annotations, output_gff, export_conf=export_conf)

    if create_library:
        fasta_library_path = f"{output_gff}.fasta"
        write_fasta_library(final_clean_annotations, fasta_file, fasta_library_path)

    end = time.time()
    print(f"⏱️ Total time: {(end - begin1):.2f} s")
    typer.secho(f"✅ Done!", fg=typer.colors.GREEN, bold=True)

if __name__ == "__main__":
    app()
