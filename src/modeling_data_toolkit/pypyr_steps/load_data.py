import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd
import psrcelmerpy

from modeling_data_toolkit import Pipeline

TABLE_READERS = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
}
SPATIAL_SUFFIXES = {".shp", ".gdb"}


def _file_group(path):
    """A shapefile is several sibling files sharing a stem; anything else is just itself."""
    if path.suffix.lower() == ".shp":
        return [p for p in path.parent.glob(f"{path.stem}.*") if p.is_file()]
    return [path]


def _mtime(path):
    """Latest modification time of a file, shapefile (all parts), or .gdb folder."""
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
    else:
        files = _file_group(path)
    return max((p.stat().st_mtime for p in files), default=path.stat().st_mtime)


def _newest_remote(file_name, remote_dirs):
    """Return the most recently modified existing copy of file_name, or None."""
    candidates = [Path(d) / file_name for d in remote_dirs if d and (Path(d) / file_name).exists()]
    return max(candidates, key=_mtime, default=None)


def _copy(remote, local_path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if remote.is_dir():
        if local_path.exists():
            shutil.rmtree(local_path)
        shutil.copytree(remote, local_path)
    else:
        for src in _file_group(remote):
            shutil.copy2(src, local_path.with_name(src.name))  # copy2 keeps mtime for later comparisons


def _sync_to_data_dir(local_path, remote_dirs, force_download=False):
    """Make sure local_path holds the right copy of the file or folder, and return it.

    - Not in data_dir: copy from the newest of network_dir / onedrive_dir.
    - In data_dir and a remote copy is newer: copy the remote copy over it.
    - force_download: always copy from the newest remote copy.
    """
    remote = _newest_remote(local_path.name, remote_dirs)

    if not local_path.exists():
        if remote is None:
            raise FileNotFoundError(
                f"{local_path.name} not found in data_dir, network_dir, or onedrive_dir"
            )
        needs_copy = True
    elif force_download:
        if remote is None:
            raise FileNotFoundError(
                f"force_download set but {local_path.name} not found in network_dir or onedrive_dir"
            )
        needs_copy = True
    else:
        needs_copy = remote is not None and _mtime(remote) > _mtime(local_path)

    if needs_copy:
        _copy(remote, local_path)
        print(f"Copied {remote}")
    return local_path


def _resolve_force(item, force_download):
    """Per-entry force_download in the config overrides the call-level flag."""
    return item.get("force_download", force_download)


def _stage_file(pipeline, item, force_download=False):
    """Resolve an input_data entry to a local path, copying it into data_dir if needed."""
    local_path = Path(pipeline.get_data_path()) / item["file"]
    onedrive_dir = item.get("onedrive_dir")
    remote_dirs = [
        item.get("network_dir"),
        Path(pipeline.get_onedrive_path()) / onedrive_dir if onedrive_dir else None,
    ]
    return _sync_to_data_dir(
        local_path,
        remote_dirs=remote_dirs,
        force_download=_resolve_force(item, force_download),
    )


def _read_spatial_layer(path, item):
    if path.suffix.lower() == ".gdb":
        layer = item.get("gdb_layer")
        if not layer:
            raise ValueError(f"'{item['name']}' is a .gdb and needs a 'gdb_layer' setting")
        return gpd.read_file(path, layer=layer)
    return gpd.read_file(path)

def _rename_columns(df, item):
    """Apply the optional col_rename mapping ({old_name: new_name}) from the config."""
    col_rename = item.get("col_rename")
    if not col_rename:
        return df
    missing = [c for c in col_rename if c not in df.columns]
    if missing:
        raise KeyError(f"col_rename columns not found in '{item['name']}': {missing}")
    return df.rename(columns=col_rename)

def load_tables(pipeline, tables, force_download=False):
    for table in tables or []:
        reader = TABLE_READERS.get(Path(table["file"]).suffix.lower())
        if reader is None:
            raise ValueError(f"Unsupported file type for table '{table['name']}': {table['file']}")

        local_path = _stage_file(pipeline, table, force_download)
        pipeline.save_table(table["name"], _rename_columns(reader(local_path), table))


def load_spatial_layers(pipeline, spatial_layers, force_download=False):
    for layer in spatial_layers or []:
        if Path(layer["file"]).suffix.lower() not in SPATIAL_SUFFIXES:
            raise ValueError(f"Unsupported file type for layer '{layer['name']}': {layer['file']}")

        local_path = _stage_file(pipeline, layer, force_download)
        gdf = _rename_columns(_read_spatial_layer(local_path, layer), layer)
        pipeline.save_geodataframe(layer["name"], gdf)


def _elmer_pending(items, tmp_dir, force_download):
    """Yield entries that aren't cached in tmp_dir yet, or that are forced to re-download."""
    for item in items or []:
        cached = tmp_dir / f"{item['name']}.parquet"
        if _resolve_force(item, force_download) or not cached.exists():
            yield item
        else:
            print(f"Skipping {item['name']}: already downloaded")


def load_elmer_geo_layers(pipeline, elmer_geo_layers, force_download=False):
    e_conn = None  # connect lazily so nothing is opened if every layer is cached
    for layer in _elmer_pending(elmer_geo_layers, pipeline.get_elmer_geo_tmp_dir(), force_download):
        e_conn = e_conn or psrcelmerpy.ElmerGeoConn()
        gdf = e_conn.read_geolayer(layer["sql_table"])
        gdf = gdf[layer["columns"] + [gdf._geometry_column_name]]
        pipeline.save_geodataframe(layer["name"], gdf, use_tmp_dir=True)


def load_elmer_tables(pipeline, elmer_tables, force_download=False):
    e_conn = None
    for table in _elmer_pending(elmer_tables, pipeline.get_elmer_tables_tmp_dir(), force_download):
        e_conn = e_conn or psrcelmerpy.ElmerConn()
        df = e_conn.get_table(table["schema"], table["sql_table"])
        pipeline.save_table(table["name"], df, use_tmp_dir=True)


def run_step(context):
    print("Running step: load_data...")

    # -- load settings
    p = Pipeline(settings_path=context['configs_dir'])
    cfg = p.settings.get('input_data', {})

    # -- load tables
    load_tables(p, cfg.get('tables', []))

    # -- load spatial layers
    load_spatial_layers(p, cfg.get('spatial_layers', []))

    # -- load Elmer tables and ElmerGeo layers
    load_elmer_tables(p, cfg.get('elmer_tables', []))
    load_elmer_geo_layers(p, cfg.get('elmer_geo_layers', []))