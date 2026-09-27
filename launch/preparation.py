"""Manifest-declared raw input checks and compute-node adapter execution."""
from pathlib import Path
import re
import sys

import experiment as common


def preparation_spec(spec):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("--prepare requires PyYAML in the login Python; select an existing interpreter with NKGRID_BOOTSTRAP_PYTHON") from exc
    manifest = yaml.safe_load(Path(spec["manifest"]).read_text(encoding="utf-8"))
    declaration = manifest.get("preparation")
    if not isinstance(declaration, dict):
        raise ValueError("--prepare requires a preparation declaration in the manifest")
    panels = [panel for panel in manifest["panels"] if panel["name"] == spec["panel"]]
    if len(panels) != 1:
        raise ValueError("Preparation requires exactly one declared panel")
    panel = panels[0]
    fields = {"panel": panel["name"], "outcome": panel["outcome"]}
    if declaration.get("panel_pattern"):
        match = re.fullmatch(declaration["panel_pattern"], panel["name"])
        if not match or match.groupdict().get("outcome", panel["outcome"]) != panel["outcome"]:
            raise ValueError("Panel does not match its preparation pattern and outcome")
        fields.update(match.groupdict())
    inputs = {}
    for key, value in declaration["inputs"].items():
        relative = Path(value.format(**fields))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Preparation inputs must be relative to --data-dir")
        inputs[key] = Path(spec["bootstrap"]["data_dir"]) / relative
    return declaration, panel, fields, inputs


def check_raw_inputs(spec):
    declaration, panel, fields, inputs = preparation_spec(spec)
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw input: {path}")
    return declaration, panel, fields, inputs


def prepare_data(spec):
    import yaml
    from aleatoric_nk_grid.run_panels import DEFAULTS, PRESETS

    declaration, panel, fields, inputs = check_raw_inputs(spec)
    settings = {**DEFAULTS, **PRESETS[spec["preset"]], **panel}
    models = spec["models"] or panel["models"]
    if len(set(models)) != len(models) or not set(models).issubset(panel["models"]):
        raise ValueError("--models must be a unique subset of the panel models")
    article = Path(spec["manifest"]).parent
    output = Path(spec["output"]) / "prepared"
    output.mkdir(exist_ok=False)
    argv = [sys.executable, article / declaration["adapter"]]
    if "config" in declaration:
        document = yaml.safe_load((article / declaration["config"]).read_text(encoding="utf-8"))
        document["paths"].update({key: str(path) for key, path in inputs.items()})
        document["paths"].update({key: str(output / name) for key, name in
                                 (("output_root", "work"), ("ard_root", "ard"), ("schema_root", "schema"))})
        document["outcomes"] = [panel["outcome"]]
        document["strategies"] = [fields["strategy"]]
        config = output / "adapter.yaml"
        config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        argv += ["--config", config]
    else:
        argv += ["--article-root", article, "--contract", article / declaration["contract"],
                 "--source", inputs["source"], "--output-root", output]
    argv += [argument.format(**fields) for argument in declaration.get("arguments", [])]
    argv += ["--validation-model", *models, "--min-n", str(settings["min_n"]),
             "--test-size", str(settings["test_size"]), "--seed", str(settings["seed"])]
    common.command(argv)
    schema = (output / declaration["schema"].format(**fields)).resolve()
    if not schema.is_relative_to(output.resolve()):
        raise ValueError("Prepared schema must be inside the run's prepared directory")
    spec["schema"] = str(schema)
    spec["schema_sha256"] = common.sha256(schema)
