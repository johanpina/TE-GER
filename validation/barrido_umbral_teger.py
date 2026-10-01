"""
barrido_umbral_teger.py
=======================
Automatiza el experimento de la hoja de ruta: corre TE-GER (rama
feature/confidence-classification) a varios umbrales de confianza, evalúa cada
salida contra TAIR10 con te_annotation_eval.py, y grafica la curva
precisión/recall que demuestra el arreglo de la sobre-predicción.

Pensado para correr EN EL SERVIDOR GPU (donde vive TE-GER + el modelo).

Flujo por cada umbral THR:
  1. python Te_annotator.py predict <fasta> <out_THR.gff3>
         --level <level> --aggregate-windows --te-threshold THR --export-conf
     (se omite si el GFF ya existe, para reanudar sin recomputar)
  2. evalúa con te_annotation_eval (bp-level + element-level, solo coordenadas)
  3. acumula P/R/F1, nº de anotaciones y Mb cubiertas
Al final: CSV + 2 gráficas (métricas vs umbral, y curva P-R con la rodilla).

Ejemplo:
  python barrido_umbral_teger.py \
    --teger-dir /home/rtabares/pina_tesis/TE-GER \
    --fasta     /home/rtabares/pina_tesis/arabidopsis.fasta \
    --gt        /home/rtabares/pina_tesis/TE-GER/TAIR10_GFF3_genes_transposons.gff \
    --out-dir   resultado_barrido \
    --thresholds 0.0,0.3,0.5,0.7,0.85,0.95 \
    --chr-map "1:Chr1,2:Chr2,3:Chr3,4:Chr4,5:Chr5" \
    --gt-keep-types "transposable_element,transposable_element_gene"
"""
import os, sys, csv, argparse, subprocess
import numpy as np


def parse_map(s):
    return dict(p.split(":") for p in s.split(",")) if s else None


def import_eval(eval_dir):
    sys.path.insert(0, eval_dir)
    import te_annotation_eval as ev
    return ev


def run_predict(py, teger_dir, fasta, out_gff, level, thr, extra, skip_existing):
    """Corre TE-GER predict para un umbral. Devuelve True si se generó/existe."""
    if skip_existing and os.path.exists(out_gff) and os.path.getsize(out_gff) > 0:
        print(f"   [skip] ya existe {out_gff}")
        return True
    cmd = [py, os.path.join(teger_dir, "Te_annotator.py"),
           fasta, out_gff, "--level", level,
           "--aggregate-windows", "--export-conf",
           "--no-create-library"]   # no necesitamos la librería FASTA en el barrido
    if thr is not None:
        cmd += ["--te-threshold", str(thr)]
    if extra:
        cmd += extra.split()
    print("   $", " ".join(cmd))
    r = subprocess.run(cmd, cwd=teger_dir)
    return r.returncode == 0 and os.path.exists(out_gff)


def evaluate(ev, gt_blocks, nuclear, gff, chr_map, min_overlap):
    """Métricas bp + elemento (solo coordenadas) de un GFF de TE-GER."""
    pred = ev.load_gff(gff, seqid_map=chr_map, normalize_types=False)
    pred = ev.keep_seqids(pred, nuclear)
    pb = ev.merge_features(pred, gap=0, by_type=False)
    bp = ev.basepair_metrics(gt_blocks, pb, ignore_type=True)["micro"]
    el = ev.element_metrics(gt_blocks, pb, min_overlap=min_overlap, type_aware=False)
    return {
        "n_frag": len(pred["start"]), "n_anot": len(pb["start"]),
        "Mb_TE": round((bp["tp"] + bp["fp"]) / 1e6, 2),
        "bp_P": round(bp["precision"], 4), "bp_R": round(bp["recall"], 4),
        "bp_F1": round(bp["f1"], 4),
        "el_P": round(el["precision"], 4), "el_R": round(el["recall"], 4),
        "el_F1": round(el["f1"], 4),
    }


def main():
    ap = argparse.ArgumentParser(description="Barrido de umbral de confianza de TE-GER")
    ap.add_argument("--teger-dir", required=True, help="Carpeta con Te_annotator.py (PinaNet/TE-GER)")
    ap.add_argument("--fasta", required=True, help="Genoma FASTA de entrada")
    ap.add_argument("--gt", required=True, help="GFF de referencia (TAIR10)")
    ap.add_argument("--out-dir", default="resultado_barrido")
    ap.add_argument("--level", default="superfamilies")
    ap.add_argument("--thresholds", default="0.0,0.3,0.5,0.7,0.85,0.95")
    ap.add_argument("--chr-map", default="1:Chr1,2:Chr2,3:Chr3,4:Chr4,5:Chr5")
    ap.add_argument("--gt-keep-types", default="transposable_element,transposable_element_gene")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--eval-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="Carpeta con te_annotation_eval.py")
    ap.add_argument("--python", default=sys.executable, help="Intérprete para TE-GER")
    ap.add_argument("--extra-predict-args", default="", help="Args extra para predict (p.ej. '--device cuda')")
    ap.add_argument("--skip-existing", action="store_true", help="No recomputar GFFs ya generados")
    ap.add_argument("--eval-only", action="store_true", help="Solo evaluar GFFs existentes (no correr predict)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    ev = import_eval(a.eval_dir)
    chr_map = parse_map(a.chr_map)
    keep = a.gt_keep_types.split(",") if a.gt_keep_types else None
    thrs = [float(x) for x in a.thresholds.split(",")]

    print(">> Cargando referencia...")
    gt = ev.load_gff(a.gt, keep_types=keep)
    gt_blocks = ev.merge_features(gt, gap=0, by_type=False)
    nuclear = sorted(set(gt["seqid"]))
    print(f"   GT={len(gt['start']):,} loci en {nuclear}")

    rows = []
    for thr in thrs:
        tag = f"thr{thr:.2f}"
        gff = os.path.join(a.out_dir, f"teger_{tag}.gff3")
        print(f"\n>> Umbral {thr}")
        if not a.eval_only:
            ok = run_predict(a.python, a.teger_dir, a.fasta, gff, a.level, thr,
                             a.extra_predict_args, a.skip_existing)
            if not ok:
                print(f"   !! falló predict para {thr}, se omite")
                continue
        if not os.path.exists(gff):
            print(f"   !! no existe {gff}, se omite")
            continue
        m = evaluate(ev, gt_blocks, nuclear, gff, chr_map, a.min_overlap)
        m = {"threshold": thr, **m}
        rows.append(m)
        print(f"   n_anot={m['n_anot']:,}  Mb={m['Mb_TE']}  "
              f"bp[P={m['bp_P']} R={m['bp_R']} F1={m['bp_F1']}]  "
              f"el[P={m['el_P']} R={m['el_R']} F1={m['el_F1']}]")

    if not rows:
        print("\nNo hay resultados. ¿Corriste predict o apuntaste a GFFs existentes?")
        return

    # --- CSV ---
    csv_path = os.path.join(a.out_dir, "barrido_umbral.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n>> CSV: {csv_path}")

    # rodilla = máximo F1 base
    best = max(rows, key=lambda r: r["bp_F1"])
    print(f">> Rodilla (máx F1 bp): umbral={best['threshold']} "
          f"P={best['bp_P']} R={best['bp_R']} F1={best['bp_F1']}")

    # --- Gráficas ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["threshold"] for r in rows]
    C_P, C_R, C_F = "#D85A30", "#185FA5", "#0F6E56"

    # (1) métricas vs umbral
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["bp_P"] for r in rows], "o-", color=C_P, label="Precisión (bp)")
    ax.plot(x, [r["bp_R"] for r in rows], "s-", color=C_R, label="Recall (bp)")
    ax.plot(x, [r["bp_F1"] for r in rows], "^-", color=C_F, label="F1 (bp)")
    ax.axvline(best["threshold"], color="gray", ls="--", alpha=0.6,
               label=f"rodilla (thr={best['threshold']})")
    ax.set_xlabel("umbral de confianza (--te-threshold)"); ax.set_ylabel("métrica (bp)")
    ax.set_ylim(0, 1); ax.set_title("Efecto del umbral de confianza de TE-GER")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "metricas_vs_umbral.png"), dpi=300)

    # (2) curva precisión-recall (cada punto = un umbral)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    R = [r["bp_R"] for r in rows]; P = [r["bp_P"] for r in rows]
    ax.plot(R, P, "o-", color="#534AB7")
    for r in rows:
        ax.annotate(f"{r['threshold']:.2f}", (r["bp_R"], r["bp_P"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.scatter([best["bp_R"]], [best["bp_P"]], s=120, facecolors="none",
               edgecolors="#D85A30", linewidths=2, label="máx F1", zorder=5)
    ax.set_xlabel("Recall (bp)"); ax.set_ylabel("Precisión (bp)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Curva precisión-recall vs umbral de confianza")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "curva_precision_recall.png"), dpi=300)

    print(f">> Gráficas en {a.out_dir}/: metricas_vs_umbral.png, curva_precision_recall.png")


if __name__ == "__main__":
    main()
