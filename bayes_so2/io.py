from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import DeterministicModel, DiscoveredSISSOFiles, ScreenedSubspace
from .utils import operator_count


def discover_sisso_files(sisso_dir: str | Path) -> DiscoveredSISSOFiles:
    """Discover the SISSO artifacts available in a directory."""
    root = Path(sisso_dir).resolve()
    pdf_candidates = sorted(root.glob("*.pdf"))
    manuscript_pdf = root / "P-SISSO-v0.pdf"
    if not manuscript_pdf.exists():
        manuscript_pdf = pdf_candidates[0] if pdf_candidates else None

    models_dir = root / "Models"
    ranking_candidates = sorted(
        path
        for path in models_dir.glob("top*_D*")
        if path.is_file() and not path.name.endswith("_coeff")
    )
    coeff_candidates = sorted(models_dir.glob("top*_D*_coeff"))
    descriptor_candidates = sorted(models_dir.glob("data_top*/desc_D*.dat"))

    return DiscoveredSISSOFiles(
        sisso_dir=root,
        manuscript_pdf=manuscript_pdf if manuscript_pdf and manuscript_pdf.exists() else None,
        sisso_in=root / "SISSO.in" if (root / "SISSO.in").exists() else None,
        sisso_out=root / "SISSO.out" if (root / "SISSO.out").exists() else None,
        train_dat=root / "train.dat" if (root / "train.dat").exists() else None,
        uspace_expressions=root / "SIS_subspaces" / "Uspace.expressions"
        if (root / "SIS_subspaces" / "Uspace.expressions").exists()
        else None,
        uspace_dat=root / "SIS_subspaces" / "Uspace.dat"
        if (root / "SIS_subspaces" / "Uspace.dat").exists()
        else None,
        models_ranking=ranking_candidates[0] if ranking_candidates else None,
        models_coefficients=coeff_candidates[0] if coeff_candidates else None,
        descriptor_data=descriptor_candidates[0] if descriptor_candidates else None,
    )


def parse_train_dat(path: Path) -> dict[str, Any]:
    """Parse the SISSO training data table."""
    with path.open() as handle:
        header = handle.readline().split()
        rows = [line.split() for line in handle if line.strip()]
    if len(header) < 3:
        raise ValueError(f"Unexpected train.dat header in {path}")
    sample_names = [row[0] for row in rows]
    y = np.array([float(row[1]) for row in rows], dtype=float)
    primary_features = np.array([[float(value) for value in row[2:]] for row in rows], dtype=float)
    return {
        "sample_id_label": header[0],
        "target_name": header[1],
        "primary_feature_names": header[2:],
        "sample_names": sample_names,
        "y": y,
        "primary_features": primary_features,
    }


def parse_uspace_expressions(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse screened expressions and SIS scores."""
    expressions: list[str] = []
    sis_scores: list[float] = []
    pattern = re.compile(r"^(?P<expr>.+?)\s+SIS_score\s*=\s*(?P<score>[-+0-9.Ee]+)\s*$")
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = pattern.match(stripped)
        if not match:
            raise ValueError(f"Could not parse Uspace expression line: {line}")
        expressions.append(match.group("expr").strip())
        sis_scores.append(float(match.group("score")))
    return expressions, np.array(sis_scores, dtype=float)


def parse_sisso_in(path: Path) -> dict[str, Any]:
    """Extract a few useful control values from SISSO.in."""
    metadata: dict[str, Any] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("!")[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def parse_models_ranking(path: Path) -> list[dict[str, Any]]:
    """Parse the SISSO top-model ranking file."""
    models: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^\s*(?P<rank>\d+)\s+(?P<rmse>[-+0-9.]+)\s+(?P<maxae>[-+0-9.]+)\s+\((?P<ids>[^)]+)\)\s*$"
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if not match:
            continue
        feature_ids = tuple(int(token) for token in match.group("ids").split())
        models.append(
            {
                "rank": int(match.group("rank")),
                "rmse": float(match.group("rmse")),
                "maxae": float(match.group("maxae")),
                "feature_ids": feature_ids,
            }
        )
    return models


def parse_models_coefficients(path: Path) -> dict[int, dict[str, Any]]:
    """Parse coefficient vectors for ranked SISSO models."""
    rows: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r"^\s*(\d+)\s+(.+)$")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if not match or line.startswith("Model_ID"):
            continue
        model_id = int(match.group(1))
        values = np.fromstring(match.group(2), sep=" ", dtype=float)
        if values.size < 2:
            continue
        rows[model_id] = {
            "intercept": float(values[0]),
            "coefficients": values[1:],
        }
    return rows


def parse_descriptor_data(path: Path) -> dict[str, Any]:
    """Parse the optional descriptor-value dump for the top model."""
    with path.open() as handle:
        header = handle.readline().split()
    data = np.loadtxt(path, skiprows=1)
    return {"header": header, "data": np.atleast_2d(data)}


def parse_sisso_out(path: Path) -> dict[str, Any]:
    """Recover the final deterministic descriptor from SISSO.out."""
    text = path.read_text()
    desc_dim_match = re.search(r"Descriptor dimension:\s+(\d+)", text)
    descriptor_dimension = int(desc_dim_match.group(1)) if desc_dim_match else None

    feature_pattern = re.compile(r"d\d+\s*=\s*(.*?)\s+feature_ID:(\d+)")
    features = [(expr.strip(), int(fid)) for expr, fid in feature_pattern.findall(text)]

    coeff_match = re.search(r"coeff\.\(ci\):\s+([^\n]+)", text)
    coeffs = np.fromstring(coeff_match.group(1), sep=" ", dtype=float) if coeff_match else None
    intercept_match = re.search(r"c0:\s+([^\n]+)", text)
    intercept = float(intercept_match.group(1)) if intercept_match else 0.0
    metric_match = re.search(r"RMSE and MaxAE:\s+([^\n]+)", text)
    metrics = np.fromstring(metric_match.group(1), sep=" ", dtype=float) if metric_match else None

    return {
        "descriptor_dimension": descriptor_dimension,
        "feature_ids": tuple(fid for _, fid in features) if features else tuple(),
        "feature_expressions": tuple(expr for expr, _ in features) if features else tuple(),
        "coefficients": coeffs,
        "intercept": intercept,
        "rmse": float(metrics[0]) if metrics is not None and metrics.size >= 1 else None,
        "maxae": float(metrics[1]) if metrics is not None and metrics.size >= 2 else None,
    }


def _build_deterministic_model(
    expressions: list[str],
    sisso_out: dict[str, Any] | None,
    ranking: list[dict[str, Any]],
    coefficients: dict[int, dict[str, Any]],
) -> DeterministicModel | None:
    if sisso_out and sisso_out["feature_ids"] and sisso_out["coefficients"] is not None:
        feature_ids = tuple(int(value) for value in sisso_out["feature_ids"])
        feature_expressions = tuple(expressions[idx - 1] for idx in feature_ids)
        return DeterministicModel(
            dimension=int(sisso_out["descriptor_dimension"] or len(feature_ids)),
            feature_ids=feature_ids,
            feature_expressions=feature_expressions,
            coefficients=np.array(sisso_out["coefficients"], dtype=float),
            intercept=float(sisso_out["intercept"]),
            rmse=sisso_out["rmse"],
            maxae=sisso_out["maxae"],
        )

    if ranking:
        top_model = ranking[0]
        coeff_row = coefficients.get(int(top_model["rank"]))
        if coeff_row:
            feature_ids = tuple(int(value) for value in top_model["feature_ids"])
            feature_expressions = tuple(expressions[idx - 1] for idx in feature_ids)
            return DeterministicModel(
                dimension=len(feature_ids),
                feature_ids=feature_ids,
                feature_expressions=feature_expressions,
                coefficients=np.array(coeff_row["coefficients"], dtype=float),
                intercept=float(coeff_row["intercept"]),
                rmse=float(top_model["rmse"]),
                maxae=float(top_model["maxae"]),
            )
    return None


def load_sisso_directory(sisso_dir: str | Path) -> tuple[ScreenedSubspace, DiscoveredSISSOFiles]:
    """Load the real SISSO screened subspace and deterministic model metadata."""
    files = discover_sisso_files(sisso_dir)
    required = {
        "train.dat": files.train_dat,
        "SIS_subspaces/Uspace.expressions": files.uspace_expressions,
        "SIS_subspaces/Uspace.dat": files.uspace_dat,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "Missing required SISSO files for screened-subspace analysis: " + ", ".join(missing)
        )

    train = parse_train_dat(files.train_dat)
    expressions, sis_scores = parse_uspace_expressions(files.uspace_expressions)
    raw_uspace = np.loadtxt(files.uspace_dat)
    raw_uspace = np.atleast_2d(raw_uspace)
    expected_columns = len(expressions) + 1
    if raw_uspace.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {files.uspace_dat}, found {raw_uspace.shape[1]}"
        )

    y_from_uspace = raw_uspace[:, 0]
    train_y = train["y"]
    sample_names = train["sample_names"]
    primary_features = train["primary_features"]
    train_rows_used = len(train_y)
    if y_from_uspace.shape != train_y.shape:
        if len(train_y) >= len(y_from_uspace):
            candidate_y = train_y[: len(y_from_uspace)]
            candidate_diff = float(np.max(np.abs(y_from_uspace - candidate_y)))
            if candidate_diff <= 1e-8:
                train_y = candidate_y
                sample_names = sample_names[: len(y_from_uspace)]
                primary_features = primary_features[: len(y_from_uspace), :]
                train_rows_used = len(train_y)
            else:
                raise ValueError("Mismatch between Uspace.dat row count and train.dat sample count")
        else:
            raise ValueError("Mismatch between Uspace.dat row count and train.dat sample count")
    y_mismatch = float(np.max(np.abs(y_from_uspace - train_y)))
    if y_mismatch > 1e-8:
        raise ValueError(
            f"train.dat targets and Uspace.dat first column disagree (max abs diff {y_mismatch:.3e})"
        )

    ranking = parse_models_ranking(files.models_ranking) if files.models_ranking else []
    coeffs = parse_models_coefficients(files.models_coefficients) if files.models_coefficients else {}
    sisso_in = parse_sisso_in(files.sisso_in) if files.sisso_in else {}
    sisso_out = parse_sisso_out(files.sisso_out) if files.sisso_out else None
    descriptor_data = parse_descriptor_data(files.descriptor_data) if files.descriptor_data else None

    deterministic_model = _build_deterministic_model(expressions, sisso_out, ranking, coeffs)
    complexity_vector = np.array([operator_count(expr) for expr in expressions], dtype=float)
    metadata: dict[str, Any] = {
        "files": {
            "manuscript_pdf": files.manuscript_pdf,
            "sisso_in": files.sisso_in,
            "sisso_out": files.sisso_out,
            "train_dat": files.train_dat,
            "uspace_expressions": files.uspace_expressions,
            "uspace_dat": files.uspace_dat,
            "models_ranking": files.models_ranking,
            "models_coefficients": files.models_coefficients,
            "descriptor_data": files.descriptor_data,
        },
        "sis_scores": sis_scores,
        "train_primary_feature_names": train["primary_feature_names"],
        "train_primary_features": primary_features,
        "sisso_in": sisso_in,
        "ranking_models": ranking,
        "descriptor_data": descriptor_data,
        "y_validation_max_abs_diff": y_mismatch,
        "train_rows_available": int(len(train["y"])),
        "train_rows_used": int(train_rows_used),
    }
    if deterministic_model:
        metadata["deterministic_support_feature_ids"] = deterministic_model.feature_ids

    subspace = ScreenedSubspace(
        X=raw_uspace[:, 1:],
        y=y_from_uspace,
        feature_ids=np.arange(1, len(expressions) + 1, dtype=int),
        feature_names=expressions,
        sample_names=sample_names,
        target_name=train["target_name"],
        complexity_vector=complexity_vector,
        metadata=metadata,
        deterministic_model=deterministic_model,
    )
    return subspace, files
