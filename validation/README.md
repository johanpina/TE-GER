# Validación de la anotación de TE-GER

Scripts para evaluar la anotación de TE-GER frente a una referencia curada
(p.ej. TAIR10 de *Arabidopsis*) **solo por coordenadas** (sin distinguir
superfamilia, porque las taxonomías difieren), y para barrer el umbral de
confianza de la rama `feature/confidence-classification`.

## Archivos
- `te_annotation_eval.py` — métricas a nivel base (pb) y a nivel de elemento,
  consolidación token→anotación (`merge_features`, `consolidate_resolved`),
  filtros y gráficas de distribución. Requiere numpy, matplotlib; pandas opcional.
- `barrido_umbral_teger.py` — corre TE-GER a varios umbrales, evalúa cada salida
  y grafica la curva precisión/recall. Encuentra `te_annotation_eval.py`
  automáticamente (misma carpeta).

## Evaluar un GFF ya generado
```bash
python te_annotation_eval.py --gt TAIR10_GFF3_genes_transposons.gff --pred salida_teger.gff3 --pred-chr-map "1:Chr1,2:Chr2,3:Chr3,4:Chr4,5:Chr5" --gt-keep-types "transposable_element,transposable_element_gene" --out resultado
```

## Barrido de umbral de confianza (corre TE-GER + evalúa + grafica)
```bash
python validation/barrido_umbral_teger.py --teger-dir . --fasta Arabidopsis_thaliana.TAIR10.dna.toplevel.fa --gt TAIR10_GFF3_genes_transposons.gff --out-dir resultado_barrido --thresholds 0.0,0.3,0.5,0.7,0.85,0.95 --skip-existing
```

Genera `barrido_umbral.csv`, `metricas_vs_umbral.png` y `curva_precision_recall.png`.

## Notas
- Excluye automáticamente organelos Mt/Pt (la referencia es nuclear).
- `transposon_fragment` del GT se descarta (son sub-partes anidadas de los TE).
- El nivel base (pb) es la métrica robusta; el de elemento es estricto.
