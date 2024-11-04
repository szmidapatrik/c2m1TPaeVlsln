import torch
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import numpy as np
from math import ceil

from matplotlib import pyplot as plt

class HeteroGNNRoundAnalyzer:

    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Visualization
    # --------------------------------------------------------------------------------------------

    # Analyze team win probabilities in a round
    def analyze_round(self, graphs, model, round_number: int, style: str = 'dark', model_code: str = None, fig_size=(20, 5), plt_title=None, plt_legend=True, plt_show=False, save_path: str = None, return_predictions: bool = False) -> None:
        """
        Analyze team win probabilities in a round.
        Parameters:
        - graphs: the dataset containing the match graphs.
        - model: the model to use for the analysis.
        - round_number: the round to analyze.
        - style: the plot style. Can be 'light' ('l' for short) or 'dark' ('d' for short) or 'cs'. Default is 'light'.
        - plt_title: the title of the plot. Default is None.
        - plt_legend: whether to show the plot legend. Default is True.
        - save_path: the path to save the plot. Default is None.
        """

        # Validate style
        if style not in ['light', 'l', 'dark', 'd', 'cs']:
            raise ValueError('Invalid style. Must be "light" (or "l" for short) or "dark" (or "d" for short).')

        # If the model code is not provided, use the model class name
        if model_code is not None and model_code != '':
            exec('model_code')

        IS_TEMPORAL = False

        # Check if the dataset is temporal
        if 'TemporalHeterogeneousGNN' in str(model):
            IS_TEMPORAL = True



        # Get the round data
        if IS_TEMPORAL:
            selected_round = self._EXT_get_round_data_temporal(graphs, round_number)
        else:
            selected_round = self._EXT_get_round_data(graphs, round_number)

        # Get the predictions
        if IS_TEMPORAL:
            predictions, remaining_time = self._EXT_get_round_predictions_temporal(selected_round, model)
        else:
            predictions, _, remaining_time = self._EXT_get_round_predictions(selected_round, model)

        # If return_predictions is True, return the predictions without plotting
        if return_predictions:
            return predictions

        if style in ['cs']:

            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=fig_size)

            # Proba plots
            plt.axhline(y=50, color='lightgray', linestyle='--', label='50%')
            plt.plot(np.array(remaining_time), np.array(predictions) * 100, lw=2, label='Defender team win probability')
            plt.plot(np.array(remaining_time), (1 - np.array(predictions)) * 100, lw=2, label='Attacker team win probability')


            # Other plot params
            plt.xticks(range(115 - ceil(len(selected_round)/4), 115), fontsize=8)
            plt.ylim(0, 100);
            plt.xlim(115 - len(selected_round)/4, 115);
            plt.xlabel('Remaining time (seconds)', fontsize=12)
            plt.ylabel('Win probability (%)', fontsize=12)
            plt.gca().invert_xaxis()

        if style in ['light', 'l']:

            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=fig_size)

            # Proba plots
            plt.axhline(y=50, color='gray', linestyle='--', label='50%')
            plt.plot(range(len(predictions)), np.array(predictions) * 100, color='cyan', lw=2, label='Defender team win probability')
            plt.plot(range(len(predictions)), (1 - np.array(predictions)) * 100, color='mediumvioletred', lw=2, label='Attacker team win probability')


            # Other plot params
            plt.xticks(ticks=range(0, len(remaining_time), 20), labels=[round(remaining_time[i]) for i in range(0, len(remaining_time), 20)])
            plt.ylim(0, len(predictions));
            plt.ylim(0, 100);
            plt.xlabel('Remaining time (seconds)', fontsize=12)
            plt.ylabel('Win probability (%)', fontsize=12)
        
        if style in ['dark', 'd']:

            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 1, figsize=fig_size)

            # Proba plots
            plt.axhline(y=50, color='white', linestyle='--', label='50%')
            plt.plot(range(len(predictions)), np.array(predictions) * 100, color='cyan', lw=2, label='Defender team win probability')
            plt.plot(range(len(predictions)), (1 - np.array(predictions)) * 100, color='mediumvioletred', lw=2, label='Attacker team win probability')


            # Other plot params
            plt.xticks(ticks=range(0, len(remaining_time), 20), labels=[round(remaining_time[i]) for i in range(0, len(remaining_time), 20)])
            plt.ylim(0, len(predictions));
            plt.ylim(0, 100);
            plt.xlabel('Remaining time (seconds)', fontsize=12)
            plt.ylabel('Win probability (%)', fontsize=12)

        if plt_title is not None:
            plt.title(plt_title, fontsize=14)

        if plt_legend:
            plt.legend(loc='upper left', labelspacing=1)

        if save_path is not None:
            plt.savefig(save_path)
        elif plt_show:
            plt.show()





    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------


    def _EXT_get_round_data(self, graphs, round_number: int) -> dict:

        selected_round = []

        # Select round data
        for graph in graphs:

            graph_round = round(graph.y['round'], 2)
            user_input_round = round(round_number/24, 2)

            if np.float32(graph_round) == np.float32(user_input_round):
                selected_round.append(graph)

        return selected_round

    def _EXT_get_round_predictions(self, selected_round, model) -> dict:

        selected_round_loader = DataLoader(selected_round, batch_size=1, shuffle=False)

        model.eval()
        pred = []
        rem_times = []
        targets = []

        with torch.no_grad():
            for data in selected_round_loader:
                rem_times.append(data.y['remaining_time'])
                data = data.to('cuda')
                out = model(data.x_dict, data.edge_index_dict, data.y, 1).float()
                target = torch.tensor(data.y['CT_wins']).float().to('cuda')
                pred.append(torch.sigmoid(out).float().cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = [prediction[0][0] for prediction in pred]
        rem_times = [time[0] for time in rem_times]

        remaining_times = []
        for time in rem_times:
            remaining_times.append(time * (115 + 7.98) - 7.98)

        return predictions, targets, remaining_times




    def _EXT_get_round_data_temporal(self, dyn_graphs, round_number: int) -> dict:

        selected_round = []

        # Store the last graph's remaining time, as there might be overlapping dynamic graphs
        last_graph_remaining_time = 1

        # Select round data
        for dyn_graph in dyn_graphs:

            dyn_graph_round = round(dyn_graph[0].y['round'], 2)
            user_input_round = round(round_number/24, 2)

            if np.float32(dyn_graph_round) == np.float32(user_input_round):
                dyn_graph_last_remaining_time = dyn_graph[-1].y['remaining_time']
                if (dyn_graph_last_remaining_time < last_graph_remaining_time) or \
                   (dyn_graph_last_remaining_time > last_graph_remaining_time and dyn_graph_last_remaining_time < 0.9):
                    selected_round.append(dyn_graph)
                    last_graph_remaining_time = dyn_graph[-1].y['remaining_time']
                else:
                    break


        return selected_round

    def _EXT_get_round_predictions_temporal(self, selected_round, model) -> dict:

        selected_round_loader = CSTemporalDataLoader(selected_round, batch_size=1, shuffle=False)

        model.eval()
        predictions = []
        rem_times = []

        with torch.no_grad():
            for dyn_graph  in selected_round_loader:

                for graph in dyn_graph[0]:
                    rem_times.append(graph.y['remaining_time'])
                
                out = model(dyn_graph, len(dyn_graph), len(dyn_graph[0])).float()
                predictions.extend(torch.sigmoid(out.squeeze()).float().cpu().numpy())

        remaining_times = []
        for time in rem_times:
            remaining_times.append(time * (115 + 7.98) - 7.98)

        return predictions, remaining_times












from torch_geometric.nn import HeteroConv, Linear, GATv2Conv

class HeterogeneousGNN(torch.nn.Module):

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    def __init__(self, player_dims, map_dims, dense_layers, player_attention_heads=None, map_attention_heads=None):

        super().__init__()

        if player_attention_heads is not None and len(player_dims) != len(player_attention_heads):
            raise ValueError('The length of player dimensions and player attention heads arrays must be the same.')
        if map_attention_heads is not None and len(map_dims) != len(map_attention_heads):
            raise ValueError('The length of map dimensions and map attention heads arrays must be the same.')

        self.conv_layer_number = max([len(player_dims), len(map_dims)])
        self.player_convs = len(player_dims)
        self.map_convs = len(map_dims)

        # Create convolutional layers
        self.convs = torch.nn.ModuleList()
        for conv_idx in range(self.conv_layer_number):

            layer_config = {}

            if conv_idx < len(player_dims):
                if player_attention_heads is None:
                    layer_config[('player', 'is', 'player')] = GATv2Conv((-1, -1), player_dims[conv_idx], add_self_loops=False)
                else:
                    layer_config[('player', 'is', 'player')] = GATv2Conv((-1, -1), player_dims[conv_idx], add_self_loops=False, heads=player_attention_heads[conv_idx])

            if conv_idx < len(player_dims):
                # GAT
                layer_config[('player', 'closest_to', 'map')] = GATv2Conv((-1, -1), map_dims[conv_idx], add_self_loops=False)

                
            if conv_idx < len(map_dims):

                # GAT
                if map_attention_heads is None:
                    layer_config[('map', 'connected_to', 'map')] = GATv2Conv((-1, -1), map_dims[conv_idx], add_self_loops=False)
                else:
                    layer_config[('map', 'connected_to', 'map')] = GATv2Conv((-1, -1), map_dims[conv_idx], add_self_loops=False, heads=map_attention_heads[conv_idx])
                


            conv = HeteroConv(layer_config, aggr='mean')
            self.convs.append(conv)



        # Create linear layer for the flattened input
        self.linear = Linear(-1, dense_layers[0]['input_neuron_num'])

        
        # Create dense layers based on the 'dense_layers' parameter
        dense_layers_container = []
        for layer_config in dense_layers:

            if layer_config['dropout'] == 0:
                # Add the first layer manually because it has a different input size
                dense_layers_container.append(torch.nn.Linear(layer_config['input_neuron_num'], layer_config['neuron_num']))
                
                # Add activation function if it is not None - the last layer does not have sigmoid activation function because of the BCEWithLogitsLoss
                if layer_config['activation_function'] is not None:
                    dense_layers_container.append(layer_config['activation_function'])

                # Add the rest of the layers (if there are any)
                for _ in range(layer_config['num_of_layers'] - 1):
                    dense_layers_container.append(torch.nn.Linear(layer_config['neuron_num'], layer_config['neuron_num']))

                    # Add activation function if it is not None - the last layer does not have sigmoid activation function because of the BCEWithLogitsLoss
                    if layer_config['activation_function'] is not None:
                        dense_layers_container.append(layer_config['activation_function'])
            else:
                dense_layers_container.append(torch.nn.Dropout(layer_config['dropout']))
        
        self.dense = torch.nn.Sequential(*dense_layers_container)
        





    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    def forward(self, x_dict, edge_index_dict, y, batch_size):

        # Do the convolutions
        conv_idx = 1
        for conv in self.convs:
            temp = conv(x_dict, edge_index_dict)
            
            if conv_idx < self.player_convs:
                x_dict['player'] = temp['player']

            if conv_idx < self.map_convs:
                x_dict['map'] = temp['map']
                
            x_dict = {key: torch.nn.functional.leaky_relu(x) for key, x in x_dict.items()}

            conv_idx += 1


        # Container for the flattened graphs after the convolutions
        flattened_graphs = []

        # Do the convolutions for each graph in the batch
        for graph_idx in range(batch_size):

            # Get the actual graph
            actual_x_dict, actual_edge_index_dict = self.get_actual_graph(x_dict, edge_index_dict, graph_idx, batch_size)

            # Get the graph data
            graph_data = torch.tensor([
                y['round'][graph_idx],
                y['time'][graph_idx],
                y['remaining_time'][graph_idx],
                y['CT_alive_num'][graph_idx],
                y['T_alive_num'][graph_idx],
                y['CT_total_hp'][graph_idx],
                y['T_total_hp'][graph_idx],
                y['CT_equipment_value'][graph_idx],
                y['T_equipment_value'][graph_idx],
                y['CT_losing_streak'][graph_idx],
                y['T_losing_streak'][graph_idx],
                y['is_bomb_dropped'][graph_idx],
                y['is_bomb_being_planted'][graph_idx],
                y['is_bomb_being_defused'][graph_idx],
                y['is_bomb_planted_at_A_site'][graph_idx],
                y['is_bomb_planted_at_B_site'][graph_idx],
                y['bomb_X'][graph_idx],
                y['bomb_Y'][graph_idx],
                y['bomb_Z'][graph_idx],
                y['bomb_mx_pos1'][graph_idx],
                y['bomb_mx_pos2'][graph_idx],
                y['bomb_mx_pos3'][graph_idx],
                y['bomb_mx_pos4'][graph_idx],
                y['bomb_mx_pos5'][graph_idx],
                y['bomb_mx_pos6'][graph_idx],
                y['bomb_mx_pos7'][graph_idx],
                y['bomb_mx_pos8'][graph_idx],
                y['bomb_mx_pos9'][graph_idx],
            ]).to('cuda')

            # Create the flattened input tensor and append it to the container
            x = torch.cat([torch.flatten(actual_x_dict['player']), torch.flatten(actual_x_dict['map']), torch.flatten(graph_data)])

            flattened_graphs.append(x)

        # Stack the flattened graphs
        x = torch.stack(flattened_graphs).to('cuda')

        x = self.linear(x)
        x = torch.nn.functional.leaky_relu(x)
        
        x = self.dense(x)
        
        return x
    






    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------

    def get_actual_graph(self, x_dict, edge_index_dict, graph_idx, batch_size):

        # Node feature dictionary for the actual graph
        actual_x_dict = {}

        single_player_node_size = int(x_dict['player'].shape[0] / batch_size)
        single_map_node_size = int(x_dict['map'].shape[0] / batch_size)

        actual_x_dict['player'] = x_dict['player'][graph_idx*single_player_node_size:(graph_idx+1)*single_player_node_size, :]
        actual_x_dict['map'] = x_dict['map'][graph_idx*single_map_node_size:(graph_idx+1)*single_map_node_size, :]


        # Edge index dictionary for the actual graph
        actual_edge_index_dict = {}

        single_map_to_map_edge_size = int(edge_index_dict[('map', 'connected_to', 'map')].shape[1] / batch_size)
        single_player_to_map_edge_size = int(edge_index_dict[('player', 'closest_to', 'map')].shape[1] / batch_size)

        actual_edge_index_dict[('map', 'connected_to', 'map')] = edge_index_dict[('map', 'connected_to', 'map')] \
            [:, graph_idx*single_map_to_map_edge_size:(graph_idx+1)*single_map_to_map_edge_size] \
            - graph_idx*single_map_node_size
        
        actual_edge_index_dict[('player', 'closest_to', 'map')] = edge_index_dict[('player', 'closest_to', 'map')] \
            [:, graph_idx*single_player_to_map_edge_size:(graph_idx+1)*single_player_to_map_edge_size]
        
        actual_edge_index_dict_correction_tensor = torch.tensor([single_player_node_size*graph_idx, single_map_node_size*graph_idx]).to('cuda')
        actual_edge_index_dict[('player', 'closest_to', 'map')] = actual_edge_index_dict[('player', 'closest_to', 'map')] - actual_edge_index_dict_correction_tensor.view(-1, 1)

        
        return actual_x_dict, actual_edge_index_dict
    
class CSTemporalDataLoader(DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=False, *args, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, *args, **kwargs)

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        # Optionally shuffle the dataset
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            torch.random.manual_seed(42)  # Ensure reproducibility if needed
            indices = torch.randperm(len(self._dataset)).tolist()

        # Yield batches of DTDGs
        batch = []
        for idx in indices:
            batch.append(self._dataset[idx])
            if len(batch) == self._batch_size:
                yield batch
                batch = []

        # Yield the last smaller batch if it exists
        if batch:
            yield batch
