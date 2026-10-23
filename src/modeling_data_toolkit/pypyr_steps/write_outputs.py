from pathlib import Path

import pandas as pd
import shapely

from modeling_data_toolkit import Pipeline

GEOM_FAMILY = {
    "Polygon": "polygon", "MultiPolygon": "polygon",
    "LineString": "line", "LinearRing": "line", "MultiLineString": "line",
    "Point": "point", "MultiPoint": "point",
}
MULTI_BUILDERS = {
    "polygon": shapely.MultiPolygon,
    "line": shapely.MultiLineString,
    "point": shapely.MultiPoint,
}


def _dated(name, today):
    return f"{name}_{today}"


def _flatten_geometry_collections(gdf, name):
    """Replace GeometryCollections (which a geodatabase can't store) with their parts of the
    layer's main geometry type. make_valid() produces these when it repairs a polygon."""
    is_collection = gdf.geom_type == "GeometryCollection"
    if not is_collection.any():
        return gdf

    families = gdf.geom_type[~is_collection].map(GEOM_FAMILY).dropna()
    family = families.mode().iat[0] if not families.empty else "polygon"

    def keep_main_type(geom):
        parts = [p for g in shapely.get_parts(geom) for p in shapely.get_parts(g)
                 if GEOM_FAMILY.get(p.geom_type) == family]
        return MULTI_BUILDERS[family](parts) if parts else None

    gdf = gdf.copy()
    gdf.loc[is_collection, gdf.geometry.name] = gdf.geometry[is_collection].apply(keep_main_type)
    empty = gdf.geometry.isna()
    if empty.any():
        print(f"Warning: dropping {empty.sum()} features from '{name}' with no {family} geometry")
        gdf = gdf[~empty]
    print(f"Converted {is_collection.sum()} GeometryCollection features in '{name}' to {family} geometry")
    return gdf


def save_csv_output(pipeline, table, output_dir, today):
    """Write a pipeline table to output_dir as <file stem>_<date>.csv."""
    out_path = output_dir / table["file"]
    out_path = out_path.with_name(f"{_dated(out_path.stem, today)}{out_path.suffix or '.csv'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {table['name']} to {out_path}...")
    pipeline.get_table(table["name"]).to_csv(out_path, index=False)
    return out_path


def save_spatial_output(pipeline, layer, output_dir, today):
    """Write a pipeline layer to a layer named <gdb_layer>_<date> in a file geodatabase."""
    gdb_path = output_dir / layer["file"]
    if gdb_path.suffix.lower() != ".gdb":
        raise ValueError(f"Output layer '{layer['name']}' must be a .gdb, got: {layer['file']}")
    if not layer.get("gdb_layer"):
        raise ValueError(f"Output layer '{layer['name']}' needs a 'gdb_layer' setting")
    layer_name = _dated(layer["gdb_layer"], today.replace("-", "_"))  # gdb layer names can't contain hyphens
    gdb_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving {layer_name} to {gdb_path}...")
    gdf = _flatten_geometry_collections(pipeline.get_geodataframe(layer["name"]), layer["name"])
    # writing to an existing layer name replaces that layer and leaves the others untouched
    gdf.to_file(gdb_path, layer=layer_name, driver="OpenFileGDB", promote_to_multi=True)
    return gdb_path


def run_step(context):
    # pypyr step
    p = Pipeline(settings_path=context['configs_dir'])
    cfg = p.settings.get('output_data', {})
    print("Saving outputs from pipeline to output directory...")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir = Path(p.get_output_path())

    for table in cfg.get('tables') or []:
        if not p.get_table_path(table['name']).exists():
            print(f"Warning: table '{table['name']}' not found in pipeline. Skipping.")
            continue
        save_csv_output(p, table, output_dir, today)

    for layer in cfg.get('spatial_layers') or []:
        if not p.get_table_path(layer['name']).exists():
            print(f"Warning: spatial layer '{layer['name']}' not found in pipeline. Skipping.")
            continue
        save_spatial_output(p, layer, output_dir, today)

    return context
