import pandas as pd
import yaml
from pathlib import Path
import os
import geopandas as gpd

# pandas 3.0 defaults to pyarrow-backed string dtypes ("future.infer_string").
# Parquet round-trips then hand geopandas ArrowDtype-backed string columns,
# which triggers a memory-blowup bug in geopandas.sjoin's reindexing step.
# Force classic numpy object-dtype strings to avoid that.
pd.set_option('future.infer_string', False)

class Pipeline:
    def __init__(self, settings_path='configs'):
        """
        Initialize Pipeline with settings loaded from a YAML file.
        """
        self.settings_path = Path(settings_path).resolve()
        self.base_dir = self.settings_path.parent
        self.user_onedrive = self._get_user_onedrive_path()

        with open(self.settings_path / 'settings.yaml', 'r') as file:
            self.settings = yaml.safe_load(file)

        # create data and output directories if they don't exist
        self.create_directory(path=self.get_data_path())
        self.create_directory(path=self.get_output_path())
        self.create_directory(path=self.get_pipeline_path())
        self.create_directory(path=self.get_elmer_tables_tmp_dir())
        self.create_directory(path=self.get_elmer_geo_tmp_dir())

    def create_directory(self, path_parts: list=None, path: str=None) -> Path:
        """Create a directory if it doesn't exist."""
        if path_parts:
            path = Path(os.path.join(*path_parts))
        else:
            path_parts = path

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")

    def _resolve_workspace_path(self, configured_path, default_name):
        path = Path(configured_path or default_name)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def get_settings_path(self):
        # Returns the path to the settings directory
        return str(self.settings_path)
    
    def get_data_path(self, *path_parts):
        return self._resolve_workspace_path(self.settings.get('data_dir'), 'data').joinpath(*path_parts)

    def get_pipeline_path(self, *path_parts):
        return self._resolve_workspace_path(self.settings.get('pipeline_dir'), 'pipeline').joinpath(*path_parts)

    def _get_user_onedrive_path(self) -> Path:
        home_dir = Path.home()
        candidates = [
            home_dir / 'PSRC',
            home_dir / 'Puget Sound Regional Council',
            home_dir / 'OneDrive - PSRC',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"OneDrive/PSRC directory not found under {home_dir}. "
            "Expected one of: PSRC, Puget Sound Regional Council, OneDrive - PSRC."
        )

    def get_onedrive_path(self, *path_parts):
        # Returns the path to the user's OneDrive directory
        return self.user_onedrive.joinpath(*path_parts)

    def get_output_path(self, *path_parts):
        return self._resolve_workspace_path(self.settings.get('output_dir'), 'output').joinpath(*path_parts)
    
    def get_table_path(self, table_name):
        return self.get_pipeline_path(f'{table_name}.parquet')

    def save_table(self, table_name, df, use_tmp_dir=False):
        if use_tmp_dir:
            tmp_dir = self.get_elmer_tables_tmp_dir()
            print(f"Saving table {table_name} to {tmp_dir}...")
            df.to_parquet(tmp_dir / f'{table_name}.parquet')
        else:
            print(f"Saving table {table_name} to pipeline...")
            df.to_parquet(self.get_table_path(table_name))

    def get_table(self, table_name, use_tmp_dir=False):
        if use_tmp_dir:
            tmp_dir = self.get_elmer_tables_tmp_dir()
            return pd.read_parquet(tmp_dir / f'{table_name}.parquet')
        else:
            return pd.read_parquet(self.get_table_path(table_name))

    def save_geodataframe(self, name, gdf, use_tmp_dir=False):
        # geopandas' parquet writer stores CRS in the file metadata, so it round-trips
        # correctly instead of relying on a hardcoded default when reading it back.
        if use_tmp_dir:
            tmp_dir = self.get_elmer_geo_tmp_dir()
            print(f"Saving spatial layer {name} to {tmp_dir}...")
            gdf.to_parquet(tmp_dir / f'{name}.parquet')
        else:
            print(f"Saving spatial layer {name} to pipeline...")
            gdf.to_parquet(self.get_table_path(name))

    def get_geodataframe(self, name, crs='epsg:2285', use_tmp_dir=False):
        if use_tmp_dir:
            tmp_dir = self.get_elmer_geo_tmp_dir()
            gdf = gpd.read_parquet(tmp_dir / f'{name}.parquet')
        else:
            gdf = gpd.read_parquet(self.get_table_path(name))
        # reproject (not just relabel) to the working CRS so buffer distances stay in feet
        # regardless of the native CRS the source layer was stored in
        if crs is not None and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
        # repair invalid geometry to avoid GEOS TopologyExceptions in downstream overlay/dissolve ops
        gdf['geometry'] = gdf['geometry'].make_valid()
        return gdf

    def get_elmer_geo_tmp_dir(self):
        return Path.home() / 'elmer_geo_tmp'

    def get_elmer_tables_tmp_dir(self):
        return Path.home() / 'elmer_tables_tmp'
    
    def convert_id_to_int64(self, table, df):
        if 'id_col' in table:
            id_col = table['id_col']
            df[id_col] = df[id_col].astype('int64')
            return df
        else:
            return df