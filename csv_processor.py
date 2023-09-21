import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class CSVProcessor:
    @staticmethod
    def aggregate(complete, ag):
        df = pd.read_csv(complete)
        df.drop(columns=CSVProcessor.get_geo_columns(), axis=1, inplace=True)
        spatial_columns = CSVProcessor.get_spatial_columns(df)
        columns_to_agg = df.columns.drop(spatial_columns)

        agg_dict = {}
        agg_dict["counter"] = ("som", 'count')
        agg_dict["som_std"] = ("som", 'std')
        for col in columns_to_agg:
            agg_dict[col] = (col, "mean")

        df_group_object = df.groupby(spatial_columns)
        df_mean = df_group_object.agg(**agg_dict).reset_index()
        df_mean.insert(0, "cell", df_mean.index)
        df_mean = df_mean[df_mean["counter"] >= 1]
        df_mean = df_mean.round(4)
        df_mean.to_csv(ag, index=False)

    @staticmethod
    def make_ml_ready(ag, ml):
        df = pd.read_csv(ag)
        df = CSVProcessor.make_ml_ready_df(df)
        df = df.round(4)
        df.to_csv(ml, index=False)

    @staticmethod
    def make_ml_ready_df(df):
        for col in ["when"]:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        for col in df.columns:
            if col not in ["scene","row","column","counter","som_std","cell"]:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
        return df

    @staticmethod
    def get_spatial_columns(df):
        spatial_columns = ["row", "column"]
        if "scene" in df.columns:
            spatial_columns = ["scene"] + spatial_columns
        return spatial_columns

    @staticmethod
    def get_geo_columns():
        return ["lon", "lat", "when"]

    @classmethod
    def gridify(cls, ag, grid):
        parent = os.path.dirname(grid)
        grids_folder = os.path.join(parent,"grids")
        os.mkdir(grids_folder)
        df = pd.read_csv(ag)
        dest = pd.DataFrame(columns=["sample","som"])

        sample = 1
        for index, row in df.iterrows():
            matrix = CSVProcessor.find_neighbours(index, row, df)
            if matrix is not None:
                np.save(f"{grids_folder}/{sample}.npy")


        dest = dest.round(4)
        dest.to_csv(grid, index=False)

    @staticmethod
    def find_neighbours(index, row, df):
        the_row = row["row"]
        the_column = row["column"]
        the_scene = None
        scene_fusion = ("scene" in df.columns)
        if scene_fusion:
            the_scene = row["scene"]

        matrix = np.zeros((12, 3, 3))
        row_offsets = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]
                                ])
        col_offsets = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]
                                ])

        for row_index in range(3):
            for col_index in range(3):
                target_row = the_row + row_offsets[row_index, col_index]
                target_col = the_column + col_offsets[row_index, col_index]
                if scene_fusion:
                    filter = df[(df["row"] == target_row) & (df["column"] == target_col) & (df["scene"] == the_scene)]
                else:
                    filter = df[(df["row"] == target_row) & (df["column"] == target_col)]
                if len(filter) == 0:
                    return None
                filter = filter[0]
                matrix[row_index, col_index] = filter


        if neighbours is None:
            continue

        new_row = {}
        for column in df.columns:
            new_row[column] = row[column]

        for ncol in n_cols:
            band = ncol[1:]
            new_row[ncol] = neighbours[band].mean()
