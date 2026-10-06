from __future__ import annotations

import csv, json, sys
from pathlib import Path
from .paths import REFERENCE_ROOT, normalize_mode, output_dir, result_root

FIGURE_TABLES={1:"figure1_plot_data.csv",3:"figure3_plot_data.csv",4:"figure4_plot_data.csv",5:"figure5_plot_data.csv",6:"figure6_plot_data.csv",7:"figure7_plot_data.csv"}
FIGURE_PNGS = {
    1: "figure1.png", 2: "cora_alpha_epsilon_m_panels.png", 3: "figure4_final_ldp_sim_panels.png",
    4: "figure5_final_norm_scale_curves.png", 5: "figure6_tao2_curves.png", 6: "figure1_heter.png",
    7: "figure5_heter.png", 8: "cora_pm_hds_alpha_lambda_epsilon.png",
}

def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as h: return list(csv.DictReader(h))

def _import_draw(name):
    repo=Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path: sys.path.insert(0,str(repo))
    return __import__(f"draw_figure.{name}",fromlist=[name])

def _source_table(figure_id, mode):
    mode = normalize_mode(mode)
    if mode != "reference":
        path = result_root(figure_id, mode) / "plot_data.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No generated plot data: {path}")
        return path
    return REFERENCE_ROOT/FIGURE_TABLES[figure_id]

def _show_png(path):
    data=Path(path).read_bytes()
    try:
        from IPython.display import Image, display
        display(Image(data=data, format="png"))
    except Exception:
        print(f"Generated PNG in memory: {len(data)} bytes")

def render_reference(figure_id:int, *, mode="reference", destination=None):
    mode = normalize_mode(mode)
    directory = Path(destination) if destination is not None else output_dir(mode, figure_id)
    directory.mkdir(parents=True, exist_ok=True)
    source = None
    if figure_id in (2,8):
        import subprocess
        script=Path(__file__).resolve().parents[2]/"draw_figure"/f"draw_figure{figure_id}.py"
        subprocess.run([sys.executable,str(script),"--output-dir",str(directory)],check=True)
    else:
        if mode in {"scaled", "full"}:
            from .search_results import validate_search_output
            validate_search_output(figure_id, mode)
        elif mode == "fixed":
            from .fixed_results import validate_fixed_output
            validate_fixed_output(figure_id)
        source = _source_table(figure_id, mode)
        module=_import_draw(f"draw_figure{figure_id}"); rows=_rows(source)
        if figure_id in (1, 6):
            datasets = tuple(name for name in module.DATASETS if any(row["dataset"] == name for row in rows))
            backbones = tuple(name for name in module.BACKBONES if any(row["backbone"] == name for row in rows))
            if figure_id == 1:
                module.plot_panels(rows, directory, datasets=datasets, backbones=backbones)
            else:
                if mode == "reference":
                    module.validate_plot_rows(rows)
                module.save_figure(rows, directory, datasets=datasets, backbones=backbones)
        elif figure_id == 3:
            module.validate_plot_rows(rows); module.plot_panels(rows, directory)
        elif figure_id in (4, 5, 7):
            if mode != "reference":
                baseline_values = {float(row["featfree_acc"]) for row in rows}
                if len(baseline_values) != 1:
                    raise RuntimeError(f"Inconsistent FeatFree baseline for Figure {figure_id}")
                baseline = baseline_values.pop()
            else:
                baseline = _load_metrics()["flickr_featfree" if figure_id == 7 else "cora_featfree"]
            module.plot_curves(rows, baseline, directory)
    _show_png(directory / FIGURE_PNGS[figure_id])
    result={"figure_id":figure_id,"mode":mode,"displayed":True,"source_table":str(source) if source else None,"output_dir":str(directory)}
    print(json.dumps(result,indent=2)); return result

def _load_metrics():
    import yaml
    with (REFERENCE_ROOT/"reference_metrics.yaml").open(encoding="utf-8") as h: return {k:float(v) for k,v in yaml.safe_load(h).items()}
