"""
te_annotation_eval.py
=====================
Evaluación del desempeño de TE-GER (PinaNet) a NIVEL DE ANOTACIÓN frente a
una referencia (ground truth) manual / EDTA, y gráficas descriptivas de la
distribución de las anotaciones.

Reemplaza al script `compare_gffs` (O(N*M), greedy, solapamiento asimétrico)
por un esquema vectorizado y metodológicamente más sólido.

----------------------------------------------------------------------------
POR QUÉ EL SCRIPT ORIGINAL ES LENTO Y SESGADO
----------------------------------------------------------------------------
1. Complejidad O(P*G): doble bucle Python sobre cada predicción y cada
   elemento del GT. Con anotaciones genómicas (10^5-10^7 features) son horas.
2. Solapamiento ASIMÉTRICO: `overlap / (p.end - p.start)` solo mide cuánto de
   la PREDICCIÓN cae en el GT. Una predicción diminuta dentro de un TE enorme
   cuenta como TP perfecto; un TE enorme apenas tocado por una predicción
   también. Lo correcto es solapamiento recíproco o IoU.
3. `break` en la PRIMERA coincidencia >= umbral, no en la MEJOR. El match
   depende del orden del archivo.
4. Denominadores inconsistentes: TP cuenta predicciones, FN cuenta GT no
   emparejados. Una región del GT cubierta por 3 predicciones suma 3 TP pero
   1 GT emparejado -> infla la precisión.
5. Ignora la SUPERFAMILIA: solapar coordenadas no es acertar la clase. En
   "genomic entity recognition" el acierto es localización + tipo.

----------------------------------------------------------------------------
QUÉ HACE ESTE MÓDULO
----------------------------------------------------------------------------
Tres niveles complementarios (de más a menos estricto/robusto):

[A] NIVEL BASE / NUCLEÓTIDO (bp-level)  -> el estándar en benchmarking de TEs
    (p.ej. EDTA, Ou et al. 2019). Por superfamilia: TP/FP/FN en pares de
    bases vía intersección de intervalos. Sin ambigüedad de emparejamiento.
    O(N log N). Da sensibilidad, precisión y F1 por clase y micro/macro.
    Es el análogo, en coordenadas genómicas, del F1 a nivel de token de la
    tesis.

[B] NIVEL DE ELEMENTO / LOCUS (entity-level, estilo NER)
    Emparejamiento UNO-A-UNO por IoU descendente con solapamiento recíproco
    (y, opcionalmente, misma superfamilia). Corrige los 5 defectos de arriba.
    Ruta rápida con pyranges (NCLS, C); fallback numpy.

[C] Gráficas de distribución (longitud vs nº de copias, boxplot por
    superfamilia, histograma) por herramienta y comparativas.

Dependencias: numpy (obligatoria), matplotlib (para gráficas).
Opcionales: pandas (E/S rápida de GFF grandes), pyranges (join rápido),
seaborn (boxplots más vistosos). Todas con fallback.

Autor: asistencia para Johan S. Piña — UAM. Junio 2026.
"""

from __future__ import annotations
import os
import numpy as np

# ----------------------------------------------------------------------------
# 0. Normalización de superfamilias / clases
# ----------------------------------------------------------------------------
# TE-GER emite la clase en la columna 'type' (LTR, TIR, LINE, HELITRON, SINE...).
# EDTA/RepeatModeler suelen ponerla en 'type' y/o en attributes (Classification=).
# Ajusta este mapa para unificar vocabularios entre herramientas.

CANON_MAP = {
    # LTR
    "ltr": "LTR", "ltr_retrotransposon": "LTR", "gypsy": "LTR", "copia": "LTR",
    "rlg": "LTR", "rlc": "LTR", "rlx": "LTR", "ltr/gypsy": "LTR", "ltr/copia": "LTR",
    # TIR / DNA
    "tir": "TIR", "dna": "TIR", "dna_transposon": "TIR", "dtt": "TIR", "dtm": "TIR",
    "dth": "TIR", "dtc": "TIR", "dta": "TIR", "terminal_inverted_repeat_element": "TIR",
    # LINE / SINE / otros órdenes
    "line": "LINE", "ril": "LINE", "sine": "SINE", "ris": "SINE",
    "helitron": "HELITRON", "dhh": "HELITRON",
    "ple": "PLE", "dirs": "DIRS", "cr1": "CR1", "piggybac": "PIGGYBAC",
    "p": "P",
}

# Tipos a IGNORAR al cargar (features no-TE típicos de EDTA/anotaciones)
IGNORE_TYPES = {
    "chromosome", "contig", "region", "repeat_region", "match", "match_part",
    "target_site_duplication", "long_terminal_repeat", "gene", "mrna", "exon",
}


def canon_type(t: str) -> str:
    """Normaliza un nombre de tipo/superfamilia a un vocabulario común."""
    if t is None:
        return "UNKNOWN"
    key = str(t).strip().lower()
    return CANON_MAP.get(key, str(t).strip().upper())


# ----------------------------------------------------------------------------
# 1. Carga de GFF eficiente
# ----------------------------------------------------------------------------
# Devolvemos un dict de arrays numpy (no DataFrame): menos memoria y todo lo
# que sigue es vectorizado. Para archivos enormes, pandas con usecols/dtype es
# el lector más rápido; si no está, hay un parser manual por líneas.

GFF_COLS = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attr"]


def load_gff(path, classify_from_attr=None, keep_types=None, drop_ignore=True,
             seqid_map=None, normalize_types=True):
    """
    Carga un GFF/GFF3 a un dict de arrays: seqid, type, start, end, strand.

    classify_from_attr: nombre de clave en la columna attributes de la que
        extraer la clase (p.ej. 'Classification' para EDTA). Si None, usa col 3.
    keep_types: iterable de tipos canónicos a conservar (None = todos).
    drop_ignore: descarta tipos estructurales (IGNORE_TYPES).
    normalize_types: si True aplica canon_type (Gypsy/Copia -> LTR, etc.). Pon
        False para CONSERVAR las etiquetas originales de TE-GER (COPIA, GYPSY,
        LARD... por separado) en el GFF consolidado.
    seqid_map: armoniza nombres de cromosoma para que coincidan entre los dos
        GFF. Puede ser:
          - dict: {'Secuencia_256': 'Chr1', '1': 'Chr1', ...}
          - callable: función str->str (p.ej. lambda s: 'Chr'+s)
        IMPRESCINDIBLE si la referencia usa 'Chr1' y TE-GER usa otro nombre:
        sin esto, el solape entre cromosomas distintos es 0.
    """
    seqid, typ, start, end, strand = _read_gff_columns(path, classify_from_attr)

    if seqid_map is not None:
        if callable(seqid_map):
            seqid = np.array([seqid_map(s) for s in seqid])
        else:
            seqid = np.array([seqid_map.get(s, s) for s in seqid])

    # Normaliza tipos y coordenadas
    if normalize_types:
        typ = np.array([canon_type(t) for t in typ])
    else:
        typ = np.array([str(t).strip().upper() for t in typ])
    start = start.astype(np.int64)
    end = end.astype(np.int64)
    # GFF es 1-based inclusivo; aseguramos start <= end
    s = np.minimum(start, end)
    e = np.maximum(start, end)

    mask = np.ones(len(s), dtype=bool)
    if drop_ignore:
        low = np.array([t.lower() for t in typ])
        mask &= ~np.isin(low, list(IGNORE_TYPES))
    if keep_types is not None:
        keep = {(canon_type(t) if normalize_types else str(t).strip().upper())
                for t in keep_types}
        mask &= np.isin(typ, list(keep))

    return {
        "seqid": seqid[mask],
        "type": typ[mask],
        "start": s[mask],
        "end": e[mask],
        "strand": strand[mask],
    }


def _read_gff_columns(path, classify_from_attr):
    """Lee columnas del GFF con pandas si está disponible, si no parser manual."""
    try:
        import pandas as pd
        usecols = [0, 2, 3, 4, 6] if classify_from_attr is None else [0, 2, 3, 4, 6, 8]
        df = pd.read_csv(
            path, sep="\t", comment="#", header=None,
            usecols=usecols, dtype={0: str},
            names=[GFF_COLS[i] for i in usecols],
            engine="c",
        )
        seqid = df["seqid"].astype(str).to_numpy()
        start = df["start"].to_numpy()
        end = df["end"].to_numpy()
        strand = df["strand"].astype(str).to_numpy()
        if classify_from_attr is not None:
            typ = _extract_attr(df["attr"].astype(str).to_numpy(), classify_from_attr)
        else:
            typ = df["type"].astype(str).to_numpy()
        return seqid, typ, start, end, strand
    except ImportError:
        return _read_gff_manual(path, classify_from_attr)


def _read_gff_manual(path, classify_from_attr):
    seqid, typ, start, end, strand = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            seqid.append(f[0]); start.append(f[3]); end.append(f[4]); strand.append(f[6])
            if classify_from_attr is not None and len(f) >= 9:
                typ.append(_attr_value(f[8], classify_from_attr))
            else:
                typ.append(f[2])
    return (np.array(seqid), np.array(typ),
            np.array(start, dtype=np.int64), np.array(end, dtype=np.int64),
            np.array(strand))


def _attr_value(attr, key):
    for field in attr.replace(";", " ").split():
        if field.lower().startswith(key.lower() + "="):
            return field.split("=", 1)[1]
        if field.lower().startswith(key.lower() + " "):
            return field.split(None, 1)[1].strip('"')
    return "UNKNOWN"


def _extract_attr(attrs, key):
    return np.array([_attr_value(a, key) for a in attrs])


# ----------------------------------------------------------------------------
# 2. Utilidades de intervalos (numpy, vectorizadas)
# ----------------------------------------------------------------------------

def _merge_intervals(start, end, gap=0):
    """
    Funde intervalos solapados/adyacentes. start,end ordenados a la salida.
    gap: distancia máxima (pb) entre dos bloques para fusionarlos. gap=0 funde
        solo solapados o pegados; gap=100 también une fragmentos separados <=100 pb.
    """
    if len(start) == 0:
        return start, end
    order = np.argsort(start, kind="mergesort")
    s = start[order]; e = end[order]
    # arranque de un nuevo bloque cuando el start supera el max-end acumulado
    cummax_end = np.maximum.accumulate(e)
    prev_max = np.empty_like(cummax_end)
    prev_max[0] = -1
    prev_max[1:] = cummax_end[:-1]
    new_block = s > prev_max + 1 + gap  # +1: adyacentes (1-based) se funden
    new_block[0] = False                # el primer intervalo abre el bloque 0
    grp = np.cumsum(new_block)
    n = grp[-1] + 1
    out_s = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    out_e = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(out_s, grp, s)
    np.maximum.at(out_e, grp, e)
    return out_s, out_e


def _total_length(start, end):
    """Suma de longitudes (pb) tras fundir solapamientos. 1-based inclusivo."""
    s, e = _merge_intervals(start, end)
    return int(np.sum(e - s + 1))


def merge_features(d, gap=0, by_type=False):
    """
    Consolida fragmentos contiguos/solapados en anotaciones completas. Es el
    paso TOKEN -> ANOTACIÓN para TE-GER, que emite predicciones por token
    (muchos fragmentos cortos, a veces duplicados con distinta etiqueta).

    by_type=False: funde TODO en bloques de 'TE' (ignora la clase). Útil para
        la comparación por coordenadas frente a la referencia.
    by_type=True: funde SOLO fragmentos de la MISMA superfamilia (preserva el
        tipo). Es la consolidación que produce la anotación final de TE-GER:
        une los tokens LTR contiguos en un LTR, los COPIA en un COPIA, etc.
        (Nota: dos clases distintas en la misma región quedan como bloques
        solapados separados — refleja la ambigüedad real del modelo.)

    gap: une fragmentos separados por <= gap pb. gap=0 funde solo solapados o
        pegados; sube a 100-2000 para puentear la fragmentación interna.

    Devuelve un dict nuevo (ordenado por seqid, start).
    """
    out_seq, out_typ, out_s, out_e = [], [], [], []
    seqs = sorted(set(d["seqid"]))
    for sq in seqs:
        msq = d["seqid"] == sq
        groups = sorted(set(d["type"][msq])) if by_type else [None]
        for g in groups:
            m = msq & (d["type"] == g) if by_type else msq
            s, e = _merge_intervals(d["start"][m], d["end"][m], gap=gap)
            out_seq.append(np.full(len(s), sq))
            out_typ.append(np.full(len(s), g if by_type else "TE"))
            out_s.append(s); out_e.append(e)
    if not out_s:
        empty = np.array([], dtype=np.int64)
        return {"seqid": np.array([]), "type": np.array([]),
                "start": empty, "end": empty, "strand": np.array([])}
    seqid = np.concatenate(out_seq)
    typ = np.concatenate(out_typ)
    start = np.concatenate(out_s)
    end = np.concatenate(out_e)
    order = np.lexsort((start, seqid))  # ordena por cromosoma y luego por inicio
    return {"seqid": seqid[order], "type": typ[order],
            "start": start[order], "end": end[order],
            "strand": np.full(len(start), ".")}


def filter_min_length(d, min_len):
    """Descarta features (fragmentos o anotaciones) más cortos que min_len pb."""
    if min_len <= 0:
        return d
    L = d["end"] - d["start"] + 1
    return _subset(d, L >= min_len)


def keep_seqids(d, seqids):
    """Conserva solo los cromosomas indicados (p.ej. excluir Mt/Pt)."""
    return _subset(d, np.isin(d["seqid"], list(seqids)))


def consolidate_resolved(d, gap=0, min_frag_len=0, min_annot_len=0, vote="bp"):
    """
    TOKEN -> ANOTACIÓN con RESOLUCIÓN DE CONFLICTOS DE CLASE.

    A diferencia de merge_features(by_type=True), que deja bloques de clases
    distintas SOLAPADOS, aquí cada región (locus) recibe UNA sola clase y las
    anotaciones resultantes NO se solapan. Pasos:

      1. (opcional) descarta fragmentos crudos < min_frag_len (ruido de tokens
         aislados de 3-7 pb por volteos de etiqueta entre ventanas solapadas).
      2. agrupa TODOS los fragmentos solapados/contiguos (<= gap) en un locus,
         SIN mirar la clase (componentes conexas por coordenadas).
      3. vota la clase del locus:
            vote='bp'    -> gana la clase con más pb dentro del locus
            vote='count' -> gana la clase con más fragmentos
      4. (opcional) descarta anotaciones finales < min_annot_len.

    Devuelve un dict de anotación NO solapante, una clase por locus.
    """
    d = filter_min_length(d, min_frag_len)
    out_seq, out_typ, out_s, out_e = [], [], [], []
    for sq in sorted(set(d["seqid"])):
        m = d["seqid"] == sq
        s = d["start"][m]; e = d["end"][m]; t = d["type"][m]
        if len(s) == 0:
            continue
        order = np.argsort(s, kind="mergesort")
        s = s[order]; e = e[order]; t = t[order]
        # componentes conexas por coordenadas (mismo criterio que _merge_intervals)
        cummax = np.maximum.accumulate(e)
        prev = np.empty_like(cummax); prev[0] = -1; prev[1:] = cummax[:-1]
        new_block = s > prev + 1 + gap
        new_block[0] = False
        blk = np.cumsum(new_block)
        nblk = int(blk[-1]) + 1
        # límites del locus (unión de sus fragmentos)
        bs = np.full(nblk, np.iinfo(np.int64).max, dtype=np.int64)
        be = np.full(nblk, np.iinfo(np.int64).min, dtype=np.int64)
        np.minimum.at(bs, blk, s)
        np.maximum.at(be, blk, e)
        # voto de clase por locus
        classes, codes = np.unique(t, return_inverse=True)
        w = (e - s + 1).astype(np.float64) if vote == "bp" else np.ones(len(s))
        tally = np.zeros((nblk, len(classes)), dtype=np.float64)
        np.add.at(tally, (blk, codes), w)
        win = classes[np.argmax(tally, axis=1)]
        out_seq.append(np.full(nblk, sq)); out_typ.append(win)
        out_s.append(bs); out_e.append(be)
    if not out_s:
        empty = np.array([], dtype=np.int64)
        return {"seqid": np.array([]), "type": np.array([]),
                "start": empty, "end": empty, "strand": np.array([])}
    seqid = np.concatenate(out_seq); typ = np.concatenate(out_typ)
    start = np.concatenate(out_s); end = np.concatenate(out_e)
    order = np.lexsort((start, seqid))
    res = {"seqid": seqid[order], "type": typ[order],
           "start": start[order], "end": end[order],
           "strand": np.full(len(start), ".")}
    return filter_min_length(res, min_annot_len)


def write_gff(d, path, source="TEGER_merged"):
    """Escribe un dict de anotación a un GFF3 (1-based, inclusivo)."""
    with open(path, "w") as fh:
        fh.write("##gff-version 3\n")
        for i in range(len(d["start"])):
            sq = d["seqid"][i]; t = d["type"][i]
            s = int(d["start"][i]); e = int(d["end"][i])
            st = d["strand"][i] if "strand" in d else "."
            fh.write(f"{sq}\t{source}\t{t}\t{s}\t{e}\t.\t{st}\t.\t"
                     f"ID={t}_{sq}_{s}_{e}\n")
    return path


def consolidation_summary(raw, merged):
    """Tabla por superfamilia: nº de fragmentos crudos -> nº de anotaciones."""
    rows = []
    classes = sorted(set(raw["type"]).union(set(merged["type"])))
    for c in classes:
        rm = raw["type"] == c
        mm = merged["type"] == c
        n_frag = int(rm.sum())
        n_annot = int(mm.sum())
        bp = int(np.sum(merged["end"][mm] - merged["start"][mm] + 1)) if n_annot else 0
        med = float(np.median(merged["end"][mm] - merged["start"][mm] + 1)) if n_annot else 0
        rows.append({"superfamily": c, "fragmentos": n_frag, "anotaciones": n_annot,
                     "factor": round(n_frag / n_annot, 1) if n_annot else 0,
                     "long_mediana": med, "pb_totales": bp})
    return rows


def _intersection_length(a_s, a_e, b_s, b_e):
    """pb de intersección entre dos conjuntos de intervalos (ya 'merged')."""
    a_s, a_e = _merge_intervals(a_s, a_e)
    b_s, b_e = _merge_intervals(b_s, b_e)
    i = j = 0
    total = 0
    na, nb = len(a_s), len(b_s)
    while i < na and j < nb:
        lo = max(a_s[i], b_s[j])
        hi = min(a_e[i], b_e[j])
        if hi >= lo:
            total += hi - lo + 1
        if a_e[i] < b_e[j]:
            i += 1
        else:
            j += 1
    return total


# ----------------------------------------------------------------------------
# 3. [A] MÉTRICAS A NIVEL DE BASE / NUCLEÓTIDO  (recomendado, estándar TE)
# ----------------------------------------------------------------------------

def basepair_metrics(gt, pred, per_seqid=True, ignore_type=True):
    """
    Precisión/sensibilidad/F1 en pares de bases (SOLO por coordenadas).

    ignore_type=True (por defecto): colapsa todas las superfamilias en una
        única clase 'TE'. Compara dónde hay TE vs dónde no, sin importar el
        tipo. Úsalo cuando la referencia manual y TE-GER usan taxonomías
        distintas y no son comparables a nivel de clase.
    ignore_type=False: desglosa por superfamilia (requiere vocabularios
        compatibles entre ambos GFF).

    Para cada clase c:
        TP = pb anotadas como c por AMBOS (intersección)
        FP = pb anotadas c por pred pero no por gt
        FN = pb anotadas c por gt pero no por pred
    No requiere emparejar elementos -> sin ambigüedad. O(N log N).

    Devuelve un dict {clase: {tp,fp,fn,precision,recall,f1,...}} más
    'micro' (global) y 'macro' (promedio simple entre clases).
    """
    if ignore_type:
        gt = {**gt, "type": np.full(len(gt["start"]), "TE")}
        pred = {**pred, "type": np.full(len(pred["start"]), "TE")}
    classes = sorted(set(gt["type"]).union(set(pred["type"])))
    results = {}
    tot_tp = tot_fp = tot_fn = 0

    for c in classes:
        gm = gt["type"] == c
        pm = pred["type"] == c
        # Intersección debe respetar el cromosoma -> por seqid
        tp = 0
        gt_len = 0
        pr_len = 0
        if per_seqid:
            seqs = set(gt["seqid"][gm]).union(set(pred["seqid"][pm]))
            for sq in seqs:
                gsel = gm & (gt["seqid"] == sq)
                psel = pm & (pred["seqid"] == sq)
                tp += _intersection_length(gt["start"][gsel], gt["end"][gsel],
                                           pred["start"][psel], pred["end"][psel])
            gt_len = _total_length_grouped(gt, gm)
            pr_len = _total_length_grouped(pred, pm)
        else:
            tp = _intersection_length(gt["start"][gm], gt["end"][gm],
                                      pred["start"][pm], pred["end"][pm])
            gt_len = _total_length(gt["start"][gm], gt["end"][gm])
            pr_len = _total_length(pred["start"][pm], pred["end"][pm])

        fp = pr_len - tp
        fn = gt_len - tp
        results[c] = _prf(tp, fp, fn, extra={"gt_bp": gt_len, "pred_bp": pr_len})
        tot_tp += tp; tot_fp += fp; tot_fn += fn

    results["micro"] = _prf(tot_tp, tot_fp, tot_fn)
    f1s = [results[c]["f1"] for c in classes]
    results["macro"] = {
        "precision": float(np.mean([results[c]["precision"] for c in classes])),
        "recall": float(np.mean([results[c]["recall"] for c in classes])),
        "f1": float(np.mean(f1s)),
    }
    return results


def _total_length_grouped(d, mask):
    seqs = d["seqid"][mask]
    tot = 0
    for sq in set(seqs):
        sel = mask & (d["seqid"] == sq)
        tot += _total_length(d["start"][sel], d["end"][sel])
    return tot


def _prf(tp, fp, fn, extra=None):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    out = {"tp": int(tp), "fp": int(fp), "fn": int(fn),
           "precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# 4. [B] MÉTRICAS A NIVEL DE ELEMENTO / LOCUS  (estilo NER, uno-a-uno)
# ----------------------------------------------------------------------------

def _overlap_pairs_numpy(a_s, a_e, b_s, b_e):
    """
    Pares de índices (ia, ib) de intervalos que se solapan, dentro de un mismo
    cromosoma. Vectorizado por ventana de candidatos (searchsorted) + filtro.
    Eficiente cuando los solapamientos son locales (caso típico de TEs).
    """
    order = np.argsort(b_s, kind="mergesort")
    bs = b_s[order]; be = b_e[order]
    ia_list, ib_list = [], []
    hi_all = np.searchsorted(bs, a_e, side="right")  # b_start <= a_end
    for i in range(len(a_s)):
        hi = hi_all[i]
        if hi == 0:
            continue
        cand = np.nonzero(be[:hi] >= a_s[i])[0]  # b_end >= a_start
        if cand.size:
            ia_list.append(np.full(cand.size, i))
            ib_list.append(order[cand])
    if not ia_list:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(ia_list), np.concatenate(ib_list)


def element_metrics(gt, pred, min_overlap=0.5, type_aware=False, metric="reciprocal"):
    """
    Empareja elementos UNO-A-UNO y calcula TP/FP/FN a nivel de elemento.

    min_overlap: umbral de solapamiento para considerar un match.
    type_aware: por defecto False -> un match SOLO depende de las coordenadas
        (un elemento cuenta si está en la posición correcta, sin importar el
        tipo). Útil cuando la referencia manual y TE-GER usan categorías
        distintas. Si True, además exige misma superfamilia.
    metric:
        'reciprocal' -> exige cobertura >= umbral en AMBOS sentidos (recíproco)
        'iou'        -> Jaccard >= umbral
    Emparejamiento codicioso por solape descendente: evita doble conteo.
    Denominadores consistentes: TP+FN = nº de GT, TP+FP = nº de predicciones.
    """
    # Construye pares candidatos por cromosoma (y por clase si type_aware)
    cand_g, cand_p, cand_score = [], [], []
    seqs = set(gt["seqid"]).union(set(pred["seqid"]))
    for sq in seqs:
        gsel = np.nonzero(gt["seqid"] == sq)[0]
        psel = np.nonzero(pred["seqid"] == sq)[0]
        if gsel.size == 0 or psel.size == 0:
            continue
        ig, ip = _overlap_pairs_numpy(gt["start"][gsel], gt["end"][gsel],
                                      pred["start"][psel], pred["end"][psel])
        if ig.size == 0:
            continue
        g_idx = gsel[ig]; p_idx = psel[ip]
        gs = gt["start"][g_idx]; ge = gt["end"][g_idx]
        ps = pred["start"][p_idx]; pe = pred["end"][p_idx]
        inter = np.minimum(ge, pe) - np.maximum(gs, ps) + 1
        inter = np.clip(inter, 0, None)
        len_g = ge - gs + 1
        len_p = pe - ps + 1
        if metric == "iou":
            score = inter / (len_g + len_p - inter)
            ok = score >= min_overlap
        else:  # reciprocal
            cov_g = inter / len_g
            cov_p = inter / len_p
            ok = (cov_g >= min_overlap) & (cov_p >= min_overlap)
            score = np.minimum(cov_g, cov_p)
        if type_aware:
            ok &= gt["type"][g_idx] == pred["type"][p_idx]
        if ok.any():
            cand_g.append(g_idx[ok]); cand_p.append(p_idx[ok]); cand_score.append(score[ok])

    n_gt = len(gt["start"]); n_pred = len(pred["start"])
    if not cand_g:
        return _element_summary(0, n_pred, n_gt)

    g_idx = np.concatenate(cand_g)
    p_idx = np.concatenate(cand_p)
    score = np.concatenate(cand_score)

    # Emparejamiento codicioso uno-a-uno por score descendente
    order = np.argsort(-score, kind="mergesort")
    used_g = np.zeros(n_gt, dtype=bool)
    used_p = np.zeros(n_pred, dtype=bool)
    tp = 0
    for k in order:
        gi, pi = g_idx[k], p_idx[k]
        if not used_g[gi] and not used_p[pi]:
            used_g[gi] = True; used_p[pi] = True
            tp += 1
    return _element_summary(tp, n_pred, n_gt, type_aware, min_overlap, metric)


def _element_summary(tp, n_pred, n_gt, type_aware=None, thr=None, metric=None):
    fp = n_pred - tp
    fn = n_gt - tp
    out = _prf(tp, fp, fn)
    out.update({"n_pred": int(n_pred), "n_gt": int(n_gt),
                "type_aware": type_aware, "threshold": thr, "metric": metric})
    return out


def element_metrics_by_class(gt, pred, min_overlap=0.5, metric="reciprocal"):
    """Métricas de elemento desglosadas por superfamilia (type_aware)."""
    classes = sorted(set(gt["type"]).union(set(pred["type"])))
    rows = {}
    for c in classes:
        g = _subset(gt, gt["type"] == c)
        p = _subset(pred, pred["type"] == c)
        rows[c] = element_metrics(g, p, min_overlap=min_overlap,
                                  type_aware=False, metric=metric)
    return rows


def _subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


# ----------------------------------------------------------------------------
# 5. [B'] Ruta rápida opcional con pyranges (para genomas completos)
# ----------------------------------------------------------------------------

def to_pyranges(d):
    import pyranges as pr
    import pandas as pd
    return pr.PyRanges(pd.DataFrame({
        "Chromosome": d["seqid"], "Start": d["start"] - 1, "End": d["end"],
        "Type": d["type"],
    }))


def element_metrics_pyranges(gt, pred, min_overlap=0.5, type_aware=False):
    """Igual que element_metrics pero usando el join O(N log N) de pyranges."""
    import pyranges as pr  # noqa
    g = to_pyranges(gt); p = to_pyranges(pred)
    j = g.join(p, suffix="_pred")  # pares solapantes
    df = j.df
    if df.empty:
        return _element_summary(0, len(pred["start"]), len(gt["start"]))
    inter = np.minimum(df.End, df.End_pred) - np.maximum(df.Start, df.Start_pred)
    len_g = df.End - df.Start
    len_p = df.End_pred - df.Start_pred
    cov_g = inter / len_g
    cov_p = inter / len_p
    ok = (cov_g >= min_overlap) & (cov_p >= min_overlap)
    if type_aware:
        ok &= df.Type.values == df.Type_pred.values
    df = df[ok]
    # Conteo uno-a-uno aproximado: GT y pred únicos emparejados
    tp = min(df.index.nunique(), len(df))
    # (para conteo exacto uno-a-uno usar element_metrics numpy)
    return _element_summary(len(df.drop_duplicates(subset=["Chromosome", "Start", "End"])),
                            len(pred["start"]), len(gt["start"]),
                            type_aware, min_overlap, "reciprocal")


# ----------------------------------------------------------------------------
# 6. Resúmenes legibles
# ----------------------------------------------------------------------------

def print_report(name_gt, name_pred, bp, elem):
    line = "-" * 64
    print(f"\n{line}\nEVALUACIÓN DE ANOTACIÓN (SOLO COORDENADAS): "
          f"{name_pred}  vs  {name_gt} (ref)\n{line}")
    print("\n[A] NIVEL BASE (pb) — TE vs no-TE, sin distinguir superfamilia")
    print(f"{'':<12}{'P':>8}{'R':>8}{'F1':>8}{'TP_bp':>12}{'FP_bp':>12}{'FN_bp':>12}")
    m = bp["micro"]
    print(f"{'TE':<12}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
          f"{m['tp']:>12}{m['fp']:>12}{m['fn']:>12}")

    print("\n[B] NIVEL ELEMENTO (uno-a-uno por posición, sin tipo)")
    print(f"  TP={elem['tp']}  FP={elem['fp']}  FN={elem['fn']}  "
          f"P={elem['precision']:.3f}  R={elem['recall']:.3f}  F1={elem['f1']:.3f}")
    print(f"  (n_pred={elem['n_pred']}  n_gt={elem['n_gt']}  "
          f"umbral={elem['threshold']}  métrica={elem['metric']})")


# ----------------------------------------------------------------------------
# 7. [C] GRÁFICAS DE DISTRIBUCIÓN
# ----------------------------------------------------------------------------

def _lengths(d):
    return d["end"] - d["start"] + 1


def plot_length_vs_count(datasets, labels, out=None, bins=60, log=True):
    """
    Longitud de secuencia (pb) vs nº de copias de esa longitud.
    datasets: lista de dicts cargados con load_gff. labels: nombres.
    Usa binning logarítmico (las longitudes de TEs abarcan varios órdenes).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    all_len = np.concatenate([_lengths(d) for d in datasets])
    all_len = all_len[all_len > 0]
    if log:
        edges = np.logspace(np.log10(all_len.min()), np.log10(all_len.max()), bins)
    else:
        edges = np.linspace(all_len.min(), all_len.max(), bins)
    centers = np.sqrt(edges[:-1] * edges[1:]) if log else (edges[:-1] + edges[1:]) / 2
    for d, lab in zip(datasets, labels):
        h, _ = np.histogram(_lengths(d), bins=edges)
        ax.plot(centers, h, marker="o", ms=3, lw=1.2, alpha=0.8, label=lab)
    if log:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Longitud del elemento (pb)")
    ax.set_ylabel("Nº de copias")
    ax.set_title("Longitud vs nº de copias")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  guardado: {out}")
    return fig


def plot_length_histogram(datasets, labels, out=None, bins=60, log=True):
    """Histograma de longitudes (uno por herramienta, superpuesto)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    all_len = np.concatenate([_lengths(d) for d in datasets])
    all_len = all_len[all_len > 0]
    edges = (np.logspace(np.log10(all_len.min()), np.log10(all_len.max()), bins)
             if log else np.linspace(all_len.min(), all_len.max(), bins))
    for d, lab in zip(datasets, labels):
        ax.hist(_lengths(d), bins=edges, alpha=0.5, label=lab)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel("Longitud del elemento (pb)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de longitudes")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  guardado: {out}")
    return fig


def plot_boxplot_by_superfamily(d, label="", out=None, log=True, min_n=1):
    """Boxplot de longitudes por superfamilia para una herramienta."""
    import matplotlib.pyplot as plt
    classes = sorted(set(d["type"]))
    data, names = [], []
    for c in classes:
        L = _lengths(d)[d["type"] == c]
        if len(L) >= min_n:
            data.append(L); names.append(f"{c}\n(n={len(L)})")
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(data)), 5))
    try:
        import seaborn as sns
        sns.boxplot(data=data, ax=ax, showfliers=False)
    except ImportError:
        ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, len(names) + 1) if not _seaborn_present() else range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Longitud (pb)")
    ax.set_title(f"Longitud por superfamilia — {label}")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  guardado: {out}")
    return fig


def plot_boxplot_compare(datasets, labels, out=None, log=True, min_n=1):
    """Boxplot comparativo: longitudes por superfamilia, agrupado por herramienta."""
    import matplotlib.pyplot as plt
    classes = sorted(set().union(*[set(d["type"]) for d in datasets]))
    classes = [c for c in classes
               if any((d["type"] == c).sum() >= min_n for d in datasets)]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(classes)), 5))
    n_tools = len(datasets)
    width = 0.8 / n_tools
    colors = plt.cm.tab10(np.linspace(0, 1, n_tools))
    for ti, (d, lab) in enumerate(zip(datasets, labels)):
        positions, data = [], []
        for ci, c in enumerate(classes):
            L = _lengths(d)[d["type"] == c]
            if len(L) >= 1:
                data.append(L)
                positions.append(ci + (ti - (n_tools - 1) / 2) * width)
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        showfliers=False, patch_artist=True)
        for box in bp["boxes"]:
            box.set(facecolor=colors[ti], alpha=0.6)
        ax.plot([], [], color=colors[ti], lw=6, alpha=0.6, label=lab)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel("Longitud (pb)")
    ax.set_title("Longitud por superfamilia — comparativo")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  guardado: {out}")
    return fig


def _seaborn_present():
    try:
        import seaborn  # noqa
        return True
    except ImportError:
        return False


def summarize_distribution(d, label=""):
    """Tabla resumen: por superfamilia n, longitud media/mediana, pb totales."""
    classes = sorted(set(d["type"]))
    L = _lengths(d)
    rows = []
    for c in classes:
        m = d["type"] == c
        Lc = L[m]
        rows.append({
            "tool": label, "superfamily": c, "n_copias": int(m.sum()),
            "long_media": float(np.mean(Lc)), "long_mediana": float(np.median(Lc)),
            "long_min": int(np.min(Lc)), "long_max": int(np.max(Lc)),
            "pb_totales": int(np.sum(Lc)),
        })
    return rows


# ----------------------------------------------------------------------------
# 8. Pipeline de conveniencia
# ----------------------------------------------------------------------------

def evaluate(gt_file, pred_file, out_dir="eval_out",
             gt_classify_attr=None, min_overlap=0.5, metric="reciprocal",
             make_plots=True, name_gt="GT", name_pred="TE-GER",
             gt_seqid_map=None, pred_seqid_map=None,
             merge_pred=True, merge_gt=False, merge_gap=0, gt_keep_types=None):
    """
    Carga, calcula métricas SOLO POR COORDENADAS (sin distinguir superfamilia)
    y genera las gráficas de distribución.

    Las métricas ignoran el tipo: un elemento/par de bases acierta si está en
    la posición correcta. Esto es lo apropiado cuando la referencia manual y
    TE-GER usan taxonomías distintas. Las gráficas SÍ desglosan por
    superfamilia, pero cada herramienta con su propio vocabulario (descriptivo,
    no comparativo).
    """
    os.makedirs(out_dir, exist_ok=True)
    gt = load_gff(gt_file, classify_from_attr=gt_classify_attr, seqid_map=gt_seqid_map,
                  keep_types=gt_keep_types)
    pred = load_gff(pred_file, seqid_map=pred_seqid_map)

    # Diagnóstico de tipos del GT: revela si se colaron features no-TE
    gt_types, gt_counts = np.unique(gt["type"], return_counts=True)
    top = sorted(zip(gt_types, gt_counts), key=lambda x: -x[1])[:8]
    print(f"Tipos en {name_gt} (top): " +
          ", ".join(f"{t}={c}" for t, c in top))
    if gt_keep_types is None:
        print(f"  (si hay tipos no-TE, filtra con gt_keep_types / --gt-keep-types)")

    # Diagnóstico: los cromosomas deben coincidir o el solape será 0
    gt_seqs, pred_seqs = set(gt["seqid"]), set(pred["seqid"])
    common = gt_seqs & pred_seqs
    print(f"Cargados: {name_gt}={len(gt['start'])} features, "
          f"{name_pred}={len(pred['start'])} features")
    print(f"Cromosomas en común: {len(common)}  "
          f"(ref={sorted(gt_seqs)[:5]}{'...' if len(gt_seqs)>5 else ''}, "
          f"pred={sorted(pred_seqs)[:5]}{'...' if len(pred_seqs)>5 else ''})")
    if not common:
        print("  *** ADVERTENCIA: 0 cromosomas en común -> todas las métricas "
              "serán 0. Usa seqid_map para armonizar los nombres. ***")

    # --- Consolidación TOKEN -> ANOTACIÓN ---
    pred_eval, gt_eval = pred, gt
    if merge_pred:
        # (1) Por clase: une fragmentos de la MISMA superfamilia -> anotación
        #     final de TE-GER. Se escribe a GFF y se resume por superfamilia.
        pred_annot = merge_features(pred, gap=merge_gap, by_type=True)
        out_gff = os.path.join(out_dir, f"{name_pred}_consolidado.gff3")
        write_gff(pred_annot, out_gff)
        print(f"\n  Consolidación por clase (token -> anotación), gap={merge_gap}:")
        print(f"    {len(pred['start'])} fragmentos -> {len(pred_annot['start'])} "
              f"anotaciones  ->  {out_gff}")
        print(f"    {'superfamilia':<14}{'fragmentos':>12}{'anotaciones':>13}"
              f"{'factor':>9}{'long_med':>10}{'pb_tot':>14}")
        for r in consolidation_summary(pred, pred_annot):
            print(f"    {r['superfamily']:<14}{r['fragmentos']:>12}{r['anotaciones']:>13}"
                  f"{r['factor']:>9}{r['long_mediana']:>10.0f}{r['pb_totales']:>14}")
        # (2) Sin clase: colapsa todo a 'TE' para la comparación por coordenadas
        pred_eval = merge_features(pred, gap=merge_gap, by_type=False)
        print(f"\n    Para la comparación por coordenadas se colapsan en "
              f"{len(pred_eval['start'])} bloques de TE.")
    if merge_gt:
        gt_eval = merge_features(gt, gap=merge_gap, by_type=False)
        print(f"  {name_gt}: {len(gt['start'])} -> {len(gt_eval['start'])} bloques de TE")

    bp = basepair_metrics(gt_eval, pred_eval, ignore_type=True)
    elem = element_metrics(gt_eval, pred_eval, min_overlap=min_overlap,
                           type_aware=False, metric=metric)
    print_report(name_gt, name_pred, bp, elem)

    if make_plots:
        # Las gráficas usan los datos ORIGINALES (con tipo) para describir la
        # distribución de longitudes por superfamilia de cada herramienta.
        ds, labs = [gt, pred], [name_gt, name_pred]
        plot_length_vs_count(ds, labs, out=os.path.join(out_dir, "longitud_vs_copias.png"))
        plot_length_histogram(ds, labs, out=os.path.join(out_dir, "histograma_longitudes.png"))
        plot_boxplot_by_superfamily(pred, label=name_pred,
                                    out=os.path.join(out_dir, f"boxplot_{name_pred}.png"))
        plot_boxplot_by_superfamily(gt, label=name_gt,
                                    out=os.path.join(out_dir, f"boxplot_{name_gt}.png"))

    return {"bp": bp, "element": elem, "gt": gt, "pred": pred}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluación de anotación TE-GER vs referencia")
    ap.add_argument("--gt", required=True, help="GFF de referencia (ground truth / EDTA)")
    ap.add_argument("--pred", required=True, help="GFF de TE-GER")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--gt-attr", default=None,
                    help="clave de attributes para la clase del GT (p.ej. Classification)")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--metric", choices=["reciprocal", "iou"], default="reciprocal")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--gt-chr-map", default=None,
                    help="renombra cromosomas del GT. Formato: 'orig1:nuevo1,orig2:nuevo2'")
    ap.add_argument("--pred-chr-map", default=None,
                    help="renombra cromosomas de TE-GER (mismo formato)")
    ap.add_argument("--merge-gap", type=int, default=0,
                    help="une fragmentos separados <= N pb al consolidar bloques de TE")
    ap.add_argument("--no-merge-pred", action="store_true",
                    help="NO consolidar los fragmentos de TE-GER (no recomendado)")
    ap.add_argument("--merge-gt", action="store_true",
                    help="también consolidar fragmentos de la referencia")
    ap.add_argument("--gt-keep-types", default=None,
                    help="conservar solo estos tipos del GT, separados por coma "
                         "(p.ej. 'transposable_element,transposable_element_gene')")
    a = ap.parse_args()

    def parse_map(s):
        if not s:
            return None
        return dict(pair.split(":") for pair in s.split(","))

    keep = a.gt_keep_types.split(",") if a.gt_keep_types else None
    evaluate(a.gt, a.pred, out_dir=a.out, gt_classify_attr=a.gt_attr,
             min_overlap=a.min_overlap, metric=a.metric, make_plots=not a.no_plots,
             gt_seqid_map=parse_map(a.gt_chr_map), pred_seqid_map=parse_map(a.pred_chr_map),
             merge_pred=not a.no_merge_pred, merge_gt=a.merge_gt, merge_gap=a.merge_gap,
             gt_keep_types=keep)
