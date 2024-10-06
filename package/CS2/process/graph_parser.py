# awpy
from awpy import Demo

# Torch
import torch

# Data handle
import pandas as pd
import numpy as np

# Other
from joblib import load, dump
import random
import json
import sys
import os

# CS2
from CS2.graph import TabularGraphSnapshot, HeteroGraphSnapshot
from CS2.token import Tokenizer
from CS2.preprocess import Dictionary, NormalizePosition, NormalizeTabularGraphSnapshot, ImputeTabularGraphSnapshot
from CS2.visualize import HeteroGraphVisualizer


class GraphParser:

    # Normalization bool flag
    NORMALIZE: bool = None

    # CONFIG constants
    CONFIG_MAP_NORM_VALUES = None
    CONFIG_MOLOTOV_RADIUS = None
    CONFIG_MOLOTOV_RADIUS_NORM = None
    CONFIG_SMOKE_RADIUS = None
    CONFIG_SMOKE_RADIUS_NORM = None

    # PATH constants
    PATH_NODDES = None
    PATH_NODDES_NORM = None
    PATH_EDGES = None



    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(
        self, 
        normalize: bool,
        PATH_NODES: str = None,
        PATH_NODES_NORM: str = None,
        PATH_EDGES: str = None,
        CONFIG_MAP_NORM_VALUES: str = None,
        CONFIG_MOLOTOV_RADIUS: str = None,
        CONFIG_MOLOTOV_RADIUS_NORM: str = None,
        CONFIG_SMOKE_RADIUS: str = None,
        CONFIG_SMOKE_RADIUS_NORM: str = None,
    ):
        
        # Validate inputs
        self.__VALIDATION__(
            normalize,
            PATH_NODES,
            PATH_NODES_NORM,
            PATH_EDGES,
            CONFIG_MAP_NORM_VALUES,
            CONFIG_MOLOTOV_RADIUS,
            CONFIG_MOLOTOV_RADIUS_NORM,
            CONFIG_SMOKE_RADIUS,
            CONFIG_SMOKE_RADIUS_NORM
        )

        self.PATH_NODDES = PATH_NODES
        self.PATH_NODES_NORM = PATH_NODES_NORM
        self.PATH_EDGES = PATH_EDGES

        self.CONFIG_MAP_NORM_VALUES = CONFIG_MAP_NORM_VALUES
        self.CONFIG_MOLOTOV_RADIUS = CONFIG_MOLOTOV_RADIUS
        self.CONFIG_MOLOTOV_RADIUS_NORM = CONFIG_MOLOTOV_RADIUS_NORM
        self.CONFIG_SMOKE_RADIUS = CONFIG_SMOKE_RADIUS
        self.CONFIG_SMOKE_RADIUS_NORM = CONFIG_SMOKE_RADIUS_NORM

        self.NORMALIZE = normalize



    # --------------------------------------------------------------------------------------------
    # REGION: Public methods
    # --------------------------------------------------------------------------------------------

    def run(
        self,
        match_path,
        player_stats_data_path,
        missing_player_stats_data_path,
        weapon_data_path,

        ticks_per_second,
        numerical_match_id,
        num_permutations_per_round,
        build_dictionary,
    ):

        # ---------- TabulaGraphSnapshot ----------
        tg = TabularGraphSnapshot()

        df, df_dict, active_infernos, active_smokes, active_he_smokes = tg.process_match(
            match_path=match_path,
            player_stats_data_path=player_stats_data_path,
            missing_player_stats_data_path=missing_player_stats_data_path,
            weapon_data_path=weapon_data_path,

            ticks_per_second=ticks_per_second,
            numerical_match_id=numerical_match_id,
            num_permutations_per_round=num_permutations_per_round,
            build_dictionary=build_dictionary
        )

        # Impute missing values
        its = ImputeTabularGraphSnapshot()
        df = its.impute(df)

        # Map nodes dataset
        if self.NORMALIZE:
            nodes = pd.read_csv(self.PATH_NODDES_NORM)
        else:
            nodes = pd.read_csv(self.PATH_NODDES)

        # Tokenize match
        tokenizer = Tokenizer()
        df = tokenizer.tokenize_match(df, 'de_inferno', nodes)




    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------


    def __VALIDATION__(
        self, 
        normalize: bool,
        PATH_NODES: str,
        PATH_NODES_NORM: str,
        PATH_EDGES: str,
        CONFIG_MAP_NORM_VALUES: str,
        CONFIG_MOLOTOV_RADIUS: str,
        CONFIG_MOLOTOV_RADIUS_NORM: str,
        CONFIG_SMOKE_RADIUS: str,
        CONFIG_SMOKE_RADIUS_NORM: str,
    ):
        
        # Validation if normalize is set to True
        if normalize:

            # PATH_NODES_NORM validation
            if PATH_NODES_NORM == None or PATH_NODES_NORM == '':
                raise ValueError('Parameter \'PATH_NODES_NORM\' is invalid.')

            # CONFIG_MAP_NORM_VALUES validation
            if CONFIG_MAP_NORM_VALUES == None or CONFIG_MAP_NORM_VALUES == '':
                raise ValueError('Parameter \'CONFIG_MAP_NORM_VALUES\' is invalid.')
            
            # CONFIG_MOLOTOV_RADIUS_NORM validation
            if CONFIG_MOLOTOV_RADIUS_NORM == None or CONFIG_MOLOTOV_RADIUS_NORM == '':
                raise ValueError('Parameter \'CONFIG_MOLOTOV_RADIUS_NORM\' is invalid.')
        
            # CONFIG_SMOKE_RADIUS_NORM validation
            if CONFIG_SMOKE_RADIUS_NORM == None or CONFIG_SMOKE_RADIUS_NORM == '':
                raise ValueError('Parameter \'CONFIG_SMOKE_RADIUS_NORM\' is invalid.')
        

        # Validation if normalize is set to False
        else:

            # PATH_NODES validation
            if PATH_NODES == None or PATH_NODES == '':
                raise ValueError('Parameter \'PATH_NODES\' is invalid.')        
        

            # CONFIG_MOLOTOV_RADIUS validation
            if CONFIG_MOLOTOV_RADIUS == None or CONFIG_MOLOTOV_RADIUS == '':
                raise ValueError('Parameter \'CONFIG_MOLOTOV_RADIUS\' is invalid.')        

            # CONFIG_SMOKE_RADIUS validation
            if CONFIG_SMOKE_RADIUS == None or CONFIG_SMOKE_RADIUS == '':
                raise ValueError('Parameter \'CONFIG_SMOKE_RADIUS\' is invalid.')
        
        
        # Leftover: PATH_EDGES validation
        if PATH_EDGES == None or PATH_EDGES == '':
            raise ValueError('Parameter \'PATH_EDGES\' is invalid.')
