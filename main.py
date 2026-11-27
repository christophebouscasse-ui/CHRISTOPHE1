import base64
import binascii
import io
import json
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

MAX_LOAD_ROWS = 100_000
MAX_RETURN_ROWS = 200
SAMPLE_ROWS = 5

_DATASETS: dict[str, pd.DataFrame] = {}
_DATASET_INFO: dict[str, dict[str, Any]] = {}


def _dataset(ctx: agents.ToolCallbackContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _DATASETS.get(ctx.session.id)
    if df is None:
        raise agents.ToolError(
            "Aucun jeu de données n'est chargé dans cette session. "
            "Chargez d'abord le fichier CSV avec load_csv_dataset."
        )
    return df, _DATASET_INFO[ctx.session.id]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_value(column: pd.Series, value: Any) -> Any:
    if pd.api.types.is_numeric_dtype(column):
        try:
            return float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            raise agents.ToolError(
                f"La colonne est numérique: la valeur doit être un nombre, pas {value!r}."
            )
    return str(value)


class Aggregation(BaseModel):
    column: str = Field(description="Colonne sur laquelle agréger.")
    function: Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique"] = Field(
        description="Fonction d'agrégation autorisée."
    )


@agents.tool(model_access="direct")
def load_csv_dataset(
    ctx: agents.ToolCallbackContext,
    /,
    file_id: Annotated[
        str,
        Field(description="ID du fichier Google Drive, extrait du lien (ex. /file/d/<ID>/view, /open?id=<ID>, ?id=<ID>) ou ID direct."),
    ],
    file_name: Annotated[str, Field(description="Nom du fichier (champ title renvoyé par get_file_metadata ou download_file_content).")],
    csv_base64: Annotated[
        str,
        Field(description="Valeur EXACTE du champ content renvoyé par download_file_content (contenu base64). Ne pas la modifier."),
    ],
    max_rows: Annotated[int, Field(description="Nombre maximal de lignes conservées.")] = MAX_LOAD_ROWS,
) -> dict[str, Any]:
    """Charge dans la session courante un fichier CSV téléchargé depuis Google Drive, et renvoie son résumé (lignes, colonnes, échantillon)."""
    try:
        raw = base64.b64decode(csv_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise agents.ToolError(f"Le contenu reçu n'est pas du base64 valide ({exc}). Reprenez le champ content de download_file_content tel quel.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    if not text.strip():
        raise agents.ToolError("Le fichier téléchargé est vide.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception as exc:
        raise agents.ToolError(f"Impossible de lire le CSV: {exc}")
    source_rows = len(df)
    truncated = source_rows > max_rows
    if truncated:
        df = df.head(max_rows)
    _DATASETS[ctx.session.id] = df
    _DATASET_INFO[ctx.session.id] = {
        "file_id": file_id,
        "file_name": file_name,
        "source_rows": source_rows,
        "truncated": truncated,
    }
    return {
        "file_id": file_id,
        "file_name": file_name,
        "loaded_rows": len(df),
        "source_rows": source_rows,
        "truncated": truncated,
        "columns": [
            {
                "name": str(c),
                "dtype": str(df[c].dtype),
                "null_count": int(df[c].isna().sum()),
                "unique_count": int(df[c].nunique(dropna=True)),
            }
            for c in df.columns
        ],
        "sample": _records(df.head(SAMPLE_ROWS)),
    }


@agents.tool(model_access="direct")
def describe_columns(
    ctx: agents.ToolCallbackContext,
    /,
) -> dict[str, Any]:
    """Décrit chaque colonne du jeu de données chargé: type, valeurs nulles, cardinalité, statistiques (min/max/moyenne pour les nombres) et valeurs les plus fréquentes."""
    df, info = _dataset(ctx)
    columns = []
    for c in df.columns:
        s = df[c]
        entry: dict[str, Any] = {
            "column": str(c),
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            entry["min"] = _num(s.min())
            entry["max"] = _num(s.max())
            entry["mean"] = _num(s.mean())
            entry["median"] = _num(s.median())
        else:
            top = s.value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": (None if pd.isna(idx) else str(idx)), "count": int(cnt)}
                for idx, cnt in top.items()
            ]
        sample = s.dropna().head(5)
        entry["sample_values"] = [str(v) for v in sample.tolist()]
        columns.append(entry)
    return {"file_name": info["file_name"], "rows": len(df), "columns": columns}


@agents.tool(model_access="direct")
def preview_rows(
    ctx: agents.ToolCallbackContext,
    /,
    n: Annotated[int, Field(description="Nombre de lignes à renvoyer (à partir du début).", ge=1, le=100)] = 10,
) -> dict[str, Any]:
    """Renvoie les n premières lignes du jeu de données chargé."""
    df, info = _dataset(ctx)
    out = df.head(n)
    return {
        "file_name": info["file_name"],
        "total_rows": len(df),
        "returned": len(out),
        "rows": _records(out),
    }


@agents.tool(model_access="direct")
def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: Annotated[str, Field(description="Colonne à filtrer.")],
    operator: Annotated[
        Literal["eq", "ne", "gt", "ge", "lt", "le", "contains", "not_contains", "isin", "not_isin", "is_null", "not_null"],
        Field(description="Opérateur de comparaison autorisé."),
    ],
    value: Annotated[
        str | list[str] | None,
        Field(description="Valeur de comparaison (nombre ou texte). Obligatoire sauf pour is_null/not_null. Pour isin/not_isin: liste de valeurs."),
    ] = None,
    limit: Annotated[int, Field(description="Nombre maximal de lignes renvoyées.", ge=1, le=MAX_RETURN_ROWS)] = 100,
) -> dict[str, Any]:
    """Filtre les lignes du jeu de données chargé avec un opérateur fixe, et renvoie le nombre de lignes correspondantes plus un échantillon."""
    df, info = _dataset(ctx)
    if column not in df.columns:
        raise agents.ToolError(f"Colonne inconnue {column!r}. Colonnes disponibles: {', '.join(str(c) for c in df.columns)}.")
    s = df[column]
    if operator in ("is_null", "not_null"):
        mask = s.isna() if operator == "is_null" else s.notna()
    elif operator in ("isin", "not_isin"):
        if value is None:
            raise agents.ToolError("isin/not_isin exige une liste de valeurs.")
        values = value if isinstance(value, list) else [value]
        values = [_coerce_value(s, v) for v in values]
        mask = s.isin(values)
        if operator == "not_isin":
            mask = ~mask
    else:
        if value is None:
            raise agents.ToolError(f"L'opérateur {operator} exige une valeur.")
        if operator in ("contains", "not_contains"):
            needle = str(value)
            mask = s.astype(str).str.contains(needle, case=False, regex=False)
            if operator == "not_contains":
                mask = ~mask
        else:
            v = _coerce_value(s, value)
            if operator == "eq":
                mask = s == v
            elif operator == "ne":
                mask = s != v
            elif operator == "gt":
                mask = s > v
            elif operator == "ge":
                mask = s >= v
            elif operator == "lt":
                mask = s < v
            else:
                mask = s <= v
    filtered = df[mask]
    out = filtered.head(limit)
    return {
        "file_name": info["file_name"],
        "filter": {"column": column, "operator": operator, "value": value},
        "matched_rows": len(filtered),
        "total_rows": len(df),
        "returned": len(out),
        "rows": _records(out),
    }


@agents.tool(model_access="direct")
def aggregate_by_group(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="Colonnes de groupement (1 ou plusieurs).", min_length=1)],
    aggregations: Annotated[list[Aggregation], Field(description="Agrégats à calculer par groupe.", min_length=1)],
    top_n: Annotated[int, Field(description="Nombre maximal de groupes renvoyés.", ge=1, le=MAX_RETURN_ROWS)] = 100,
) -> dict[str, Any]:
    """Calcule des agrégats (count, sum, mean, median, min, max, std, nunique) par groupe de lignes du jeu de données chargé."""
    df, info = _dataset(ctx)
    unknown = [c for c in group_by if c not in df.columns]
    if unknown:
        raise agents.ToolError(
            f"Colonnes de groupement inconnues: {unknown}. Colonnes disponibles: {', '.join(str(c) for c in df.columns)}."
        )
    allowed = {"count", "sum", "mean", "median", "min", "max", "std", "nunique"}
    named: dict[str, tuple[str, str]] = {}
    for agg in aggregations:
        col = agg.column if isinstance(agg, Aggregation) else agg["column"]
        fn = agg.function if isinstance(agg, Aggregation) else agg["function"]
        if fn not in allowed:
            raise agents.ToolError(f"Fonction inconnue {fn!r}. Fonctions autorisées: {', '.join(sorted(allowed))}.")
        if col not in df.columns:
            raise agents.ToolError(
                f"Colonne d'agrégation inconnue {col!r}. Colonnes disponibles: {', '.join(str(c) for c in df.columns)}."
            )
        named[f"{col}__{fn}"] = (col, fn)
    grouped = df.groupby(list(group_by), dropna=False).agg(**named).reset_index()
    total = len(grouped)
    out = grouped.head(top_n)
    return {
        "file_name": info["file_name"],
        "group_by": group_by,
        "aggregations": [f"{a.column if isinstance(a, Aggregation) else a['column']}__{a.function if isinstance(a, Aggregation) else a['function']}" for a in aggregations],
        "total_groups": total,
        "returned": len(out),
        "rows": _records(out),
    }


INSTRUCTIONS = """Tu es un analyste de données conversationnel. L'utilisateur te donne un lien Google Drive vers un fichier CSV et te pose des questions en langage naturel sur ses données. Tu réponds en français, de façon concise, avec des chiffres réels tirés du fichier, jamais inventés.

Procédure de chargement d'un fichier :
1. Extrais l'ID du fichier du lien Drive (formes /file/d/<ID>/view, /open?id=<ID>, ?id=<ID>, ou un ID direct). Si le message ne contient qu'un nom de fichier, utilise search_files du connecteur Google Drive pour trouver l'ID.
2. Appelle get_file_metadata(file_id=...) pour vérifier le fichier (titre, type, taille).
3. Appelle download_file_content(file_id=...) pour télécharger le contenu. Pour un CSV uploadé tel quel, le champ content est en base64. Pour une Google Sheet, passe export_mime_type="text/csv".
4. Appelle load_csv_dataset avec file_id, file_name (le title) et csv_base64 = le champ content EXACT de download_file_content, non modifié.

Ensuite, pour répondre aux questions, utilise UNIQUEMENT les outils fournis :
- describe_columns : comprendre les colonnes (types, nulls, statistiques).
- preview_rows : montrer les premières lignes.
- filter_rows : filtrer des lignes (un opérateur à la fois).
- aggregate_by_group : calculer des agrégats par groupe (count, sum, mean, median, min, max, std, nunique).

Règles :
- Ne jamais exécuter de code et ne jamais inventer de chiffres : chaque réponse chiffrée doit venir d'un résultat d'outil. Si une information manque dans les données, dis-le explicitement.
- Mémoire : la session conserve tout l'historique de la conversation et le jeu de données chargé reste mémorisé pour la session. L'utilisateur peut te poser des questions de suivi sans se réexpliquer ; réponds directement en t'appuyant sur ce qui a déjà été chargé et discuté. Ne recharge le fichier avec load_csv_dataset que si l'utilisateur donne un nouveau lien, ou si un outil répond qu'aucun jeu de données n'est chargé (par exemple après un redémarrage du worker) : dans ce cas, réutilise le téléchargement déjà présent dans l'historique de la session.
- Cite les noms de colonnes exacts et les nombres tels que renvoyés par les outils. Quand un résultat est tronqué (returned < matched_rows ou total_groups), dis-le.
- Le fichier Drive est lu en lecture seule : n'appelle jamais create_file, update_file ou copy_file.
- Si la question ne porte pas sur les données (hors sujet), réponds brièvement que tu es un analyste de données CSV et ramène la conversation vers le fichier chargé."""

agent = agents.Agent(
    model="mistral-large-4",
    name="csv-data-analyst",
    description=(
        "Analyste de données conversationnel : lit un fichier CSV depuis le Google Drive de l'utilisateur "
        "via son connecteur, puis répond en langage naturel avec des chiffres réels (colonnes, filtres, agrégats par groupe). "
        "Outils fixes uniquement, sans exécution de code."
    ),
    instructions=INSTRUCTIONS,
    tools=[load_csv_dataset, describe_columns, preview_rows, filter_rows, aggregate_by_group],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(skills=False, subagents=False, background_processes=False),
    permission_policy=agents.PermissionPolicy(
        default="allow",
        rules=(
            agents.PermissionRule(
                tools=("connector_google_drive_mcp.*",),
                denylist=(
                    "connector_google_drive_mcp.create_file",
                    "connector_google_drive_mcp.update_file",
                    "connector_google_drive_mcp.copy_file",
                ),
                decision="allow",
            ),
        ),
    ),
    max_iterations=30,
)
