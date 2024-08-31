import torch
from torch_geometric.data import HeteroData

from matplotlib import pyplot as plt
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
    def visualize_snapshot(self, graph: HeteroData) -> None:
        
        # Get the data
        map_nodes = graph['map'].x[:, 1:4].numpy()
        map_edges = graph['map', 'connected_to', 'map'].edge_index.numpy()

        players = graph['player'].x[:, 0:3].numpy()
        player_edges = graph['player', 'closest_to', 'map'].edge_index.numpy()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot map nodes and edges
        for edge in map_edges.T:
            plt.plot(map_nodes[edge, 0], map_nodes[edge, 1], c='black', linewidth=0.5)
        plt.scatter(map_nodes[:, 0], map_nodes[:, 1], s=10, c='black')
        plt.scatter(graph['map'].x[graph['map'].x[:, -2] == 1][:, 1], graph['map'].x[graph['map'].x[:, -2] == 1][:, 2], s=500, c='lightcoral', alpha=0.3)
        plt.scatter(graph['map'].x[graph['map'].x[:, -1] == 1][:, 1], graph['map'].x[graph['map'].x[:, -1] == 1][:, 2], s=750, c='gray', alpha=0.3)

        # Plot players
        plt.scatter(players[:5, 0], players[:5, 1], s=15, c='lightblue')
        plt.scatter(players[5:, 0], players[5:, 1], s=15, c='gold')

        for edge in player_edges.T:
            plt.plot([players[edge[0]][0], map_nodes[edge[1]][0]], [players[edge[0]][1], map_nodes[edge[1]][1]], c='grey', linewidth=0.5)

        plt.show()

    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------

   