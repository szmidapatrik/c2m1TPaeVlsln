import torch
from torch_geometric.data import HeteroData
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal
from termcolor import colored

class TemporalHeteroGraphSnapshot:


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass 



    # --------------------------------------------------------------------------------------------
    # REGION: Public methods
    # --------------------------------------------------------------------------------------------

    def create_dynamic_graph(
        self,
        graphs: list[HeteroData]
    ):
        """
        Create a dynamic graph from a list of snapshots.
        
        Parameters:
        - graphs: the list of snapshots.
        """

        # Collect all time slices here
        time_slices = []

        # Iterate over the snapshots from which the dynamic graph is to be constructed
        for graph_data in graphs: 

            # Extract the node features
            player_features = graph_data['player']['x']  # Shape [num_players, num_features]
            map_features = graph_data['map']['x']        # Shape [num_map_nodes, num_features]

            # Edge indices (relations)
            map_map_edge_index = graph_data[('map', 'connected_to', 'map')]['edge_index']
            player_map_edge_index = graph_data[('player', 'closest_to', 'map')]['edge_index']
            player_player_edge_index = graph_data[('player', 'is', 'player')]['edge_index']

            # Time information
            time = graph_data['y']['remaining_time']

            # Grapg level features
            graph_features = graph_data['y'].copy()
            del graph_features['remaining_time']
            del graph_features['time']
            del graph_features['CT_wins']


            # Create graph snapshot at this time
            snapshot = (
                # 0. feature_dicts
                {"player": player_features, "map": map_features},

                # 1. edge_index_dicts
                {('map', 'connected_to', 'map'): map_map_edge_index, 
                ('player', 'closest_to', 'map'): player_map_edge_index},

                # 2. edge_weight_dicts 
                # Create empty tensors for edge weights as they are not used
                {torch.ones(map_map_edge_index.shape[1]), torch.ones(player_map_edge_index.shape[1])},

                # 3. target_dicts
                # Create empty tensors for node target feature as they are not used
                {torch.ones(map_features.shape[0]), torch.ones(player_features.shape[0])},

                # 4. graph_features
                graph_features,

                # 5. time_stamps
                time,  # Time step

                # 6. target
                graph_data['y']['CT_wins'],  # Label for classification

            )
            
            # Append the snapshot to the list of time slices
            time_slices.append(snapshot)


        # Create the DTDG using PyG Temporal
        dynamic_graph = DynamicHeteroGraphTemporalSignal(
            feature_dicts=[slice[0] for slice in time_slices],  # Node features for each snapshot
            edge_index_dicts=[slice[1] for slice in time_slices],
            edge_weight_dicts=[slice[2] for slice in time_slices],  # Edge weights for each snapshot
            target_dicts=[slice[3] for slice in time_slices],  # Node targets (player_stats)

            graph_features=[slice[4] for slice in time_slices],  # Graph features (round_info)
            time_stamps=[slice[5] for slice in time_slices],  # Timestamps (remaining_time)
            target=[slice[6] for slice in time_slices],  # Labels (CT_wins)
        )

        return dynamic_graph



    def process_match(
        self, 
        match_graphs: list[HeteroData],
        interval: int = 10,
        shifted_intervals: bool = False,
        parse_rate: int = 16
    ):
        """
        Process the rounds of a match and create a dynamic graph with fixed length intervals.
        Parameters:
        - match_graphs: the list of snapshots for a match.
        - interval: the number of snapshots to include in a single dynamic graph.
        - shifted_intervals: whether additional shifted intervals should be created by shifting the interval window by itnerval/2 frames. Only works if interval is even.
        - parse_rate: the time between two snapshots in tick number. Default is 16 (4 ticks per second).
        """

        # Collect all dynamic graphs here
        dynamic_graphs = []

        # Get round numbers
        rounds = self._EXT_get_round_number_list_(match_graphs)

        # Iterate over the rounds
        for round_number in rounds:

            # Get the graph snapshots of the round
            round_graphs = self._EXT_get_round_graphs_(match_graphs, round_number)

            # It is probable that the number of snapshots is not devidable by the interval number, thus drop the first n snapshots to make it devidable
            default_round_graphs = round_graphs[(len(round_graphs) % interval):]

            # Iterate over the remaining snapshots
            for snpshot_idx in range(0, len(default_round_graphs), interval):

                # Tick control variables
                first_tick = default_round_graphs[snpshot_idx].y['tick']
                last_tick = default_round_graphs[snpshot_idx + interval - 1].y['tick']
                actual_tick = default_round_graphs[snpshot_idx].y['tick']

                # Skip sequence control variable
                SKIP_SEQUENCE = False

                # VALIDATION - Check if there are missing ticks in the graph sequence
                for graph in default_round_graphs[snpshot_idx: snpshot_idx + interval]:
                    if graph.y['tick'] != actual_tick :
                        print(colored('Error:', "red", attrs=["bold"]) + f'Error: There are missing ticks in the graph sequence. The error occured while parsing match {graph.y["numerical_match_id"]} at round \
                            {graph.y["round_number"]} between ticks {first_tick}-{last_tick}. Skipping the sequence.')
                        actual_tick += parse_rate
                        SKIP_SEQUENCE = True
                        break
                    actual_tick += parse_rate

                # Create dynamic graphs and add them to the dynamic_graphs list
                if not SKIP_SEQUENCE:
                    dynamic_graph = self.create_dynamic_graph(default_round_graphs[snpshot_idx: snpshot_idx + interval])
                    dynamic_graphs.append(dynamic_graph)

        return dynamic_graphs

    



    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------

    def _EXT_get_round_number_list_(self, graphs):

        # Store the unique round numbers
        round_numbers = []

        # Check the graphs and add new round numbers
        for graph in graphs:
            if (graph.y['round'] not in round_numbers):
                round_numbers.append(graph.y['round'])

        return round_numbers
    


    def _EXT_get_round_graphs_(self, graphs, round_number):

        # Store the graphs of the round
        round_graphs = []

        # Iterate over the graphs
        for graph in graphs:

            # If it's not yet the correct round number, continue
            if graph.y['round'] != round_number and len(round_graphs) == 0:
                continue

            # If it's the correct number, add the graph to the round_graphs list
            elif graph.y['round'] == round_number:
                round_graphs.append(graph)

            # After all graphs were added, break the loop
            elif graph.y['round'] != round_number and len(round_graphs) != 0:
                break

        return round_graphs
    

