import torch
from torch_geometric.data import HeteroData

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import pandas as pd
import numpy as np

import random

class HeteroGraphVisualizer:



    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Visualization
    # --------------------------------------------------------------------------------------------

    # Visualize a heterogeneous graph snapshot
    def visualize_snapshot(self, graph: HeteroData, style='light', map_image_path: str = None) -> None:
        """
        Visualize a heterogeneous graph snapshot.
        Parameters:
        - graph: the HeteroData graph to visualize.
        - style: the plot style. Can be 'light' or 'dark'. Default is 'light'.
        - map_image_path: the path to the map image. Default is None. 
        """

        # Validate style
        if style not in ['light', 'dark']:
            raise ValueError('Invalid style. Must be "light" or "dark".')


        # Get the image
        if map_image_path is not None:
            img = mpimg.imread(map_image_path)

        # Get the data
        map_nodes = graph['map'].x[:, 1:4].numpy()
        map_edges = graph['map', 'connected_to', 'map'].edge_index.numpy()

        players = graph['player'].x[:, 0:3].numpy()
        player_edges = graph['player', 'closest_to', 'map'].edge_index.numpy()

        # Set dark background
        if style == 'dark':
            plt.style.use('dark_background')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot map image
        if style == 'light':
            ax.imshow(img, extent=[-0.07, 1.0465, -0.085, 1.030], alpha=0.5)
        elif style == 'dark':
            ax.imshow(img, extent=[-0.07, 1.0465, -0.085, 1.030], alpha=0.8)


        # Visualize map with light style
        if style == 'light':
            
            # Plot map nodes and edges
            for edge in map_edges.T:
                ax.plot(map_nodes[edge, 0], map_nodes[edge, 1], c='black', linewidth=0.5)
            ax.scatter(map_nodes[:, 0], map_nodes[:, 1], s=10, c='black')
            ax.scatter(graph['map'].x[graph['map'].x[:, -2] == 1][:, 1], graph['map'].x[graph['map'].x[:, -2] == 1][:, 2], s=500, c='firebrick', alpha=0.3)
            ax.scatter(graph['map'].x[graph['map'].x[:, -1] == 1][:, 1], graph['map'].x[graph['map'].x[:, -1] == 1][:, 2], s=750, c='dimgray', alpha=0.3)

        # Visualize map with dark style
        elif style == 'dark':

            # Plot map nodes and edges
            for edge in map_edges.T:
                ax.plot(map_nodes[edge, 0], map_nodes[edge, 1], c='white', linewidth=0.5, alpha=0.5)
            ax.scatter(map_nodes[:, 0], map_nodes[:, 1], s=10, c='white', alpha=0.5)
            ax.scatter(graph['map'].x[graph['map'].x[:, -2] == 1][:, 1], graph['map'].x[graph['map'].x[:, -2] == 1][:, 2], s=500, c='darkred', alpha=0.3)
            ax.scatter(graph['map'].x[graph['map'].x[:, -1] == 1][:, 1], graph['map'].x[graph['map'].x[:, -1] == 1][:, 2], s=750, c='lightgray', alpha=0.3)

        # Visualize players with light style
        if style == 'light':

            # Plot players
            ax.scatter(players[:5, 0], players[:5, 1], s=15, c='dodgerblue')
            ax.scatter(players[5:, 0], players[5:, 1], s=15, c='darkorange')

        # Visualize players with dark style
        elif style == 'dark':

            # Plot players
            ax.scatter(players[:5, 0], players[:5, 1], s=15, c='cyan')
            ax.scatter(players[5:, 0], players[5:, 1], s=15, c='mediumvioletred')

        for edge in player_edges.T:
            ax.plot([players[edge[0]][0], map_nodes[edge[1]][0]], [players[edge[0]][1], map_nodes[edge[1]][1]], c='grey', linewidth=0.5)

        plt.show()

    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------

   