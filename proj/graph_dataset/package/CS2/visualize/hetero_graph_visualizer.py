import torch
from torch_geometric.data import HeteroData

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import pandas as pd
import numpy as np

import random
import os

class HeteroGraphVisualizer:

    # OS path for this file
    __file_path = os.path.dirname(os.path.abspath(__file__))

    # Map image path constants
    INFERNO_LIGHT = '/radar/inferno.png'
    INFERNO_DARK  = '/radar/inferno_dark.png'



    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        self.INFERNO_LIGHT = self.__file_path + self.INFERNO_LIGHT
        self.INFERNO_DARK = self.__file_path + self.INFERNO_DARK



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Visualization
    # --------------------------------------------------------------------------------------------

    # Visualize a heterogeneous graph snapshot
    def visualize_snapshot(self, graph: HeteroData, map: str, style='light') -> None:
        """
        Visualize a heterogeneous graph snapshot.
        Parameters:
        - graph: the HeteroData graph to visualize.
        - map: the map on which the match was held.
        - style: the plot style. Can be 'light' or 'dark'. Default is 'light'.
        """

        # Validate style
        if style not in ['light', 'l', 'dark', 'd']:
            raise ValueError('Invalid style. Must be "light" (or "l" for short) or "dark" (or "d" for short).')

        # Validate map
        if map not in ['de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_anubis', 'de_ancient', 'de_vertigo']:
            raise ValueError('Invalid map. Must be one of "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_anubis", "de_ancient", "de_vertigo".')
        


        # Get the image
        img = self.__EXT_get_map_radar(map, style)

        # Get the data
        map_nodes = graph['map'].x[:, 1:4].numpy()
        map_edges = graph['map', 'connected_to', 'map'].edge_index.numpy()

        players = graph['player'].x[:, 0:3].numpy()
        player_edges = graph['player', 'closest_to', 'map'].edge_index.numpy()

        # Set background color
        if style == 'light' or style == 'l':
            plt.style.use('default')
        if style == 'dark' or style == 'd':
            plt.style.use('dark_background')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set title
        ax.set_title(f'Round: {round(graph.y['round'] * 24)}\nRemaining time: {round(graph.y['remaining_time'] * (115 + -4.78125))}', fontsize=20)

        # Plot map image
        ax.imshow(img, extent=[-0.07, 1.0465, -0.085, 1.030], alpha=0.5)


        # Visualize map with light style
        if style == 'light' or style == 'l':
            
            # Plot map nodes and edges
            for edge in map_edges.T:
                ax.plot(map_nodes[edge, 0], map_nodes[edge, 1], c='black', linewidth=0.5)
            ax.scatter(map_nodes[:, 0], map_nodes[:, 1], s=10, c='black')
            ax.scatter(graph['map'].x[graph['map'].x[:, -2] == 1][:, 1], graph['map'].x[graph['map'].x[:, -2] == 1][:, 2], s=500, c='firebrick', alpha=0.3)
            ax.scatter(graph['map'].x[graph['map'].x[:, -1] == 1][:, 1], graph['map'].x[graph['map'].x[:, -1] == 1][:, 2], s=750, c='dimgray', alpha=0.3)

        # Visualize map with dark style
        elif style == 'dark' or style == 'd':

            # Plot map nodes and edges
            for edge in map_edges.T:
                ax.plot(map_nodes[edge, 0], map_nodes[edge, 1], c='white', linewidth=0.5, alpha=0.5)
            ax.scatter(map_nodes[:, 0], map_nodes[:, 1], s=10, c='white', alpha=0.5)
            ax.scatter(graph['map'].x[graph['map'].x[:, -2] == 1][:, 1], graph['map'].x[graph['map'].x[:, -2] == 1][:, 2], s=500, c='darkred', alpha=0.3)
            ax.scatter(graph['map'].x[graph['map'].x[:, -1] == 1][:, 1], graph['map'].x[graph['map'].x[:, -1] == 1][:, 2], s=750, c='lightgray', alpha=0.3)

        # Visualize players with light style
        if style == 'light' or style == 'l':

            # Plot players
            ax.scatter(players[:5, 0], players[:5, 1], s=15, c='dodgerblue')
            ax.scatter(players[5:, 0], players[5:, 1], s=15, c='darkorange')

        # Visualize players with dark style
        elif style == 'dark' or style == 'd':

            # Plot players
            ax.scatter(players[:5, 0], players[:5, 1], s=15, c='cyan')
            ax.scatter(players[5:, 0], players[5:, 1], s=15, c='mediumvioletred')

        for edge in player_edges.T:
            ax.plot([players[edge[0]][0], map_nodes[edge[1]][0]], [players[edge[0]][1], map_nodes[edge[1]][1]], c='grey', linewidth=0.5)

        plt.show()




    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------

    # Get the map radar image
    def __EXT_get_map_radar(self, map: str, style: str):
        """
        Get the map radar image.
        Parameters:
        - map: the map on which the match was held.
        - style: the plot style. Can be 'light' or 'dark'.
        Returns:
        - the map radar image.
        """

        # Get the image
        if map == 'de_inferno':
            if style == 'light' or style == 'l':
                img = mpimg.imread(self.INFERNO_LIGHT)
            elif style == 'dark' or style == 'd':
                img = mpimg.imread(self.INFERNO_DARK)

        else:
            raise ValueError('Support for this map is not yet available.')

        return img