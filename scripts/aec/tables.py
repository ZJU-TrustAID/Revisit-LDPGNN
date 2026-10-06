from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from .paths import REFERENCE_ROOT, normalize_mode, result_root
from .table_settings import TABLE6_FEATURE_DIM, table6_feature_dims

def _read(path):
    with Path(path).open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _table_seed_rows(table, mode):
    mode = normalize_mode(mode)
    if mode != "reference":
        candidate = result_root(table, mode) / f"{table}_seed_rows.csv"
        if candidate.is_file(): return _read(candidate)
        raise FileNotFoundError(f"No generated table data for {table} mode={mode}; run the experiment first")
    return _read(REFERENCE_ROOT / f"{table}_seed_rows.csv")

def _stats(rows):
    vals=[float(r["test_acc"]) for r in rows]; mean=sum(vals)/len(vals); std=(sum((x-mean)**2 for x in vals)/(len(vals)-1))**0.5 if len(vals)>1 else 0.; return mean,std

def _display(markdown):
    try:
        from IPython.display import Markdown, display
        display(Markdown(markdown))
    except Exception: print(markdown)

def table4_summary(mode="reference"):
    from .search_results import table_datasets

    mode = normalize_mode(mode)
    ff=_table_seed_rows("table4",mode); groups=defaultdict(list)
    for r in ff: groups[(r["setting"],r["backbone"],r["dataset"])].append(r)
    datasets = table_datasets("table4", mode)
    if mode in {"scaled", "full"}:
        return [
            (backbone.upper(), setting, [_stats(groups[(setting, backbone, dataset)]) for dataset in datasets])
            for backbone in ("gcn", "sage", "gat") for setting in ("Best-LDP", "FeatFree-P")
        ]
    if mode == "fixed":
        from .fixed_results import validate_fixed_output

        validate_fixed_output(1)
        plot = _read(result_root(1, mode) / "plot_data.csv")
    else:
        plot = _read(REFERENCE_ROOT / "figure1_plot_data.csv")
    best={}
    for r in plot:
        if not r.get("pipeline","").startswith("figure3_pipeline"): continue
        if str(r.get("x_eps")) not in {"10","10.0"}: continue
        key=(r["backbone"],r["dataset"]); val=float(r.get("val_acc_mean",-1))
        if key not in best or val>best[key][0]: best[key]=(val,r)
    rows=[]; datasets=["cora","lastfm","citeseer","facebook"]
    for backbone in ["gcn","sage","gat"]:
        for setting in ["Best-LDP","FeatFree-P"]:
            cells=[]
            for dataset in datasets:
                ffmean,ffstd=_stats(groups[("FeatFree-P",backbone,dataset)]); br=best[(backbone,dataset)][1]; bmean=float(br["test_acc_mean"]); bstd=float(br["test_acc_std"]);
                cells.append((bmean,bstd) if setting=="Best-LDP" else (ffmean,ffstd))
            rows.append((backbone.upper(),setting,cells))
    return rows

def _select_table6_groups(raw, dimensions):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in raw:
        key = (row["setting"], row["backbone"], row["dataset"])
        grouped[key][int(row["feature_dim"])].append(row)
    selected = {}
    for key, by_dim in grouped.items():
        missing = set(dimensions) - by_dim.keys()
        if missing:
            raise ValueError(f"Table 6 {key} is missing dimensions: {sorted(missing)}")
        dimension = dimensions[0]
        if len(dimensions) > 1:
            dimension = min(dimensions, key=lambda dim: (
                -sum(float(row["val_acc"]) for row in by_dim[dim]) / len(by_dim[dim]), dim,
            ))
        selected[key] = (str(dimension), by_dim[dimension])
    return selected


def table6_summary(mode="reference", *, feature_dim=None):
    mode = normalize_mode(mode)
    dimensions = table6_feature_dims(TABLE6_FEATURE_DIM if feature_dim is None else feature_dim)
    if mode in {"reference", "fixed"} and dimensions != (TABLE6_FEATURE_DIM,):
        raise ValueError(f"Table 6 mode={mode} requires feature_dim={TABLE6_FEATURE_DIM}")
    selected = _select_table6_groups(_table_seed_rows("table6", mode), dimensions)
    datasets=["cora","lastfm","citeseer","facebook"]
    rows=[]
    for backbone in ["gcn","sage","gat"]:
        for setting in ["FeatFree-Kprop","FeatFree-HOA"]:
            label="Kprop" if setting.endswith("Kprop") else "HOA"; cells=[]
            for dataset in datasets:
                dimension,items=selected[(setting,backbone,dataset)]; cells.append((*_stats(items),dimension))
            rows.append((backbone.upper(),label,cells))
    return rows

def render_table(table, *, mode="reference", destination=None, feature_dim=None):
    from .search_results import table_datasets, validate_search_output

    mode = normalize_mode(mode)
    if table not in {"table4", "table6"}:
        raise ValueError(f"Unsupported table: {table}")
    if mode in {"scaled", "full"}:
        validate_search_output(table, mode, feature_dim=TABLE6_FEATURE_DIM if feature_dim is None else feature_dim)
    labels = {"cora": "Cora", "lastfm": "LastFM", "citeseer": "CiteSeer", "facebook": "Facebook"}
    datasets=[labels[name] for name in table_datasets(table, mode)]
    rows=table4_summary(mode) if table=="table4" else table6_summary(mode, feature_dim=feature_dim)
    lines=["| Backbone | Setting | "+" | ".join(datasets)+" |","|---|---|"+"---|"*len(datasets)]
    for backbone,setting,cells in rows:
        if table=="table6": values=[f"{m:.2f} ± {s:.2f}" for m,s,d in cells]
        else: values=[f"{m:.2f} ± {s:.2f}" for m,s,*rest in cells]
        lines.append(f"| {backbone} | {setting} | "+" | ".join(values)+" |")
    markdown="\n".join(lines)+"\n"; _display(markdown); return {"table":table,"mode":mode,"displayed":True,"rows":len(rows)}
