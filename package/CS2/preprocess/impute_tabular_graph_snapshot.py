import pandas as pd
import polars as pl
import numpy as np


class ImputeTabularGraphSnapshot:


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Impute the tabular snapshots
    # --------------------------------------------------------------------------------------------

    def impute(
        self, 
        df, 
        method='default'
    ):
        """
        Imputes missing values in the dataset.
        
        Parameters:
            - df: the dataset to be imputed.
            - method (optional): the method to be used for imputation. Can be 'default'. Default is 'default'.
        """

        # Check the type of the input dataframe
        if isinstance(df, pd.DataFrame):
            df = self.__impute_pandas_df(df, method)

        elif isinstance(df, pl.DataFrame):
            df = self.__impute_polars_df(df, method)
            
        else:
            raise ValueError('The input dataframe must be either a pandas or polars dataframe.')

        return df

        



    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------

    def __impute_pandas_df(self, df: pd.DataFrame, method: str) -> pd.DataFrame:

        boolean_columns = [col for col in df.columns if '_is_' in col]
        df[boolean_columns] = df[boolean_columns].astype('Int8')

        # Method: default
        if method == 'default':
            df = df.infer_objects().fillna(0)
            df = df.replace(np.inf, 0)
            df = df.replace(-np.inf, 0)
            df = df.replace('null', 0)
            df = df.replace({None: 0})

        return df
    
    def __impute_polars_df(self, df: pl.DataFrame, method: str) -> pl.DataFrame:

        # Convert boolean columns to Int8
        boolean_columns = [col for col in df.columns if '_is_' in col]
        df = df.with_columns([
            pl.col(col).cast(pl.Int8).alias(col) for col in boolean_columns
        ])

        # Method: default
        if method == 'default':
            
            df = df.fill_null(0)
            df = df.fill_nan(0)
            
            non_numeric_columns = [col for col in df.columns if '_name' in col or col == 'MATCH_ID']
            non_numeric_columns += [col for col in df.columns if df[col].dtype == pl.Boolean]
            
            df = df.with_columns([
                pl.when(pl.col(c).is_infinite()).then(0).otherwise(pl.col(c)).alias(c)
                for c in df.columns if c not in non_numeric_columns
            ])

        return df
    