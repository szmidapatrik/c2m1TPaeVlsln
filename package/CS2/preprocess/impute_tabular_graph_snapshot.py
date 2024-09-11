import pandas as pd
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

        # Method: default
        if method == 'default':
            df = df.infer_objects().fillna(0)
            df = df.replace(np.inf, 0)
            df = df.replace(-np.inf, 0)

        return df



    