from typing import Union
from awpy import Demo
import pandas as pd
import numpy as np
import random

class CS2_Tokenizer:



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    def tokenize(self, df: pd.DataFrame):
        """
        Tokenizes the given snapshots of the given dataframe.

        Parameters:
            - df: pd.DataFrame: The dataframe containing the snapshots to tokenize.
        """

        # Tokenize the positions
        self._TOKEN_positions(df)

        # Tokenize the economic data
        self._TOKEN_economic_data(df)

        return df
    


    # --------------------------------------------------------------------------------------------

    def _TOKEN_positions(self, df: pd.DataFrame, map_nodes: pd.DataFrame):
        """
        Encodes player positions in the given dataframe and returns the token.

        Parameters:
            - df: pd.DataFrame: The dataframe containing the snapshots to tokenize.
            - map_nodes: pd.DataFrame: The dataframe containing the graph nodes of the map.
        """

        # Tokenize the positions
        df['positions'] = df['positions'].apply(lambda x: self._tokenize_positions(x))

        return df
    


    # --------------------------------------------------------------------------------------------

    # Calculate closest graph node to a position
    def __EXT_closest_node_to_pos__(self, coord_x, coord_y, coord_z, nodes):
        """
        Returns the closest node to a given position.
        
        Parameters:
        - coord_x: the x coordinate of the position.
        - coord_y: the y coordinate of the position.
        - coord_z: the z coordinate of the position.
        - nodes: the nodes dataframe.
        """
        distances = np.sqrt((nodes['x'] - coord_x)**2 + (nodes['y'] - coord_y)**2 + (nodes['z'] - coord_z)**2)
        return nodes.loc[distances.idxmin(), 'node_id']