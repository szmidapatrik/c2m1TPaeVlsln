import torch
from torch_geometric.data import DataLoader, HeteroData
from torch_geometric.loader import DataLoader

import shap
shap.initjs()

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score

import pandas as pd
import numpy as np
from .snapshot_events import SnapshotEvents

from matplotlib import pyplot as plt
import seaborn as sns

from math import ceil

class HeteroGNNRoundAnalyzer:

    # Predefined previous frames for the temporal analysis
    previous_frames = [1, 4, 8, 12, 16, 20]

    graphs = None
    dyn_graphs = None

    model = None

    normalizing_dictionary = None

    round_number = None
    predictions = None

    # Event datasets
    edf_1  = None
    edf_4  = None
    edf_8  = None
    edf_12 = None
    edf_16 = None
    edf_20 = None

    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self, graphs, dyn_graphs, model, round_number, dictionary=None):

        self.graphs = graphs
        self.dyn_graphs = dyn_graphs
        self.model = model
        self.round_number = round_number

        if dictionary is not None:
            self.normalizing_dictionary = dictionary

        self.predictions = self._SHAP_EVT_predict_proba(self.dyn_graphs, self.model, self.round_number)
        



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Visualization
    # --------------------------------------------------------------------------------------------

    # Predict the winning probabilities for a round
    def predict_proba(self, style: str = 'light', model_code: str = None, fig_size=(20, 5), plt_title=None, plt_legend=True, plt_show=False, save_path: str = None, return_predictions: bool = False) -> None:
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


        selected_round = self._EXT_get_round_data_temporal(self.dyn_graphs, self.round_number)

        # Get the predictions
        predictions, remaining_time = self._EXT_get_round_predictions_temporal(selected_round, self.model)

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



    # Get SHAP values for a round
    def get_shap_values(self):

        self._SHAP_EXT_process_event_datasets()

        models = self._SHAP_train_local_models()
        explainers, shap_values = self._SHAP_get_explainers_and_shap_values(models)
        masked_shap_values = self._SHAP_mask_shap_values(shap_values)
        agg_shap_values, explainer_expected_value = self._SHAP_aggregate_shap_values(masked_shap_values, explainers)

        return agg_shap_values, explainer_expected_value



    # Plot the perdormance of the local linear models
    def plot_local_preds(self):

        self._SHAP_EXT_process_event_datasets()

        self._SHAP_train_local_models(plot_results=True, return_models=False)



    # Get feature importances of the local models
    def feature_importance(self, n=0, agg='mean'):

        self._SHAP_EXT_process_event_datasets()

        models = self._SHAP_train_local_models()

        return self._SHAP_local_model_feature_importance(models, n, agg=agg)



    # Delta win probability
    def delta_proba(self):

        self._SHAP_EXT_process_event_datasets()

        fig, ax = plt.subplots(figsize=(20, 5))
        plt.plot(self.edf_1['y_change'])
        plt.xlabel('Remaining time (seconds)', fontsize=12)
        plt.ylabel('Δ Win probability (%)', fontsize=12)


        

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



    def _SHAP_train_local_models(self, print_results=False, plot_results=False, return_models=True):

        if plot_results:
            fig, axs = plt.subplots(2, 3, figsize=(7, 5))

        models = []

        for edf_idx in range(len(self.previous_frames)):

            # Access the correct edf dynamically without using exec
            edf_name = f'edf_{self.previous_frames[edf_idx]}'
            edf = getattr(self, edf_name)

            X = edf.drop(columns=['y', 'y_change', 'round', 'round_change', 'idx']).values
            y = edf['y_change']

            # Ridge regression training
            local_model = Ridge(random_state=42, alpha=0.1)
            local_model.fit(X, y)

            # Prediction
            y_pred = local_model.predict(X)
            y_true = y

            # Metrics
            r2  = round(r2_score(y, y_pred), 4)
            mse = round(mean_squared_error(y, y_pred), 4)
            mae = round(mean_absolute_error(y, y_pred), 4)
            evs = round(explained_variance_score(y, y_pred), 4)

            if print_results:
                print(f'EDF {self.previous_frames[edf_idx]} - R2: {r2}, MSE: {mse}, MAE: {mae}, EVS: {evs}')

            if plot_results:

                # Plot results
                sns.scatterplot(x=y_true, y=y_pred, ax=axs[edf_idx//3, edf_idx%3])
                axs[edf_idx//3, edf_idx%3].set_title(f'R2 score: {r2}, MSE: {mse}\nMAE: {mae}, EVS: {evs}', fontsize=8)

                # X and Y ticks smaller size
                axs[edf_idx//3, edf_idx%3].tick_params(axis='x', labelsize=6)
                axs[edf_idx//3, edf_idx%3].tick_params(axis='y', labelsize=6)

            models.append(local_model)

        if plot_results:

            plt.tight_layout()

            axs[0, 0].set_ylabel('Predicted Change')
            axs[1, 0].set_ylabel('Predicted Change')

            axs[0, 0].set_xlabel('')
            axs[0, 1].set_xlabel('')
            axs[0, 2].set_xlabel('');
            axs[1, 0].set_xlabel('True Change')
            axs[1, 1].set_xlabel('True Change')
            axs[1, 2].set_xlabel('True Change');

            plt.show()

        if return_models:
            return models

    def _SHAP_local_model_feature_importance(self, models, n, agg='mean'):

        # Feature list
        feature_list = self.edf_1.drop(columns=['y', 'y_change', 'round', 'round_change', 'idx']).columns

        # Model coefficients
        model_coefs = np.array([model.coef_ for model in models])
        if agg == 'mean':
            model_coefs = np.mean(model_coefs, axis=0)
        elif agg == 'sum':
            model_coefs = np.sum(model_coefs, axis=0)

        # Feature importances
        coef_df = pd.DataFrame({"CT features": feature_list, "importance": model_coefs})
        if n == 0:
            return coef_df

        CT_n = coef_df.sort_values('importance', ascending=False).head(n)
        T_n = coef_df.sort_values('importance', ascending=True).head(n)

        return pd.concat([CT_n.reset_index(drop=True), T_n.reset_index(drop=True)], axis=1)

    def _SHAP_get_explainers_and_shap_values(self, models):

        explainers = []
        shap_values = []

        for idx in range(len(self.previous_frames)):

            # Data
            edf_name = f'edf_{self.previous_frames[idx]}'
            edf = getattr(self, edf_name)

            X = edf.drop(columns=['y', 'y_change', 'round', 'round_change', 'idx']).values

            # SHAP explainer inicialization
            explainer = shap.LinearExplainer(models[idx], X)
            shap_vals = explainer.shap_values(X)

            explainers.append(explainer)
            shap_values.append(shap_vals)

        return explainers, shap_values

    def _SHAP_mask_shap_values(self, shap_values):

        masked_shap_values = []

        for idx in range(len(self.previous_frames)):

            edf_name = f'edf_{self.previous_frames[idx]}'
            edf = getattr(self, edf_name)

            # Get the mask values
            mask = (edf.drop(columns=['y', 'y_change', 'round', 'round_change', 'idx']) != 0) * 1
            # Get the column names
            mask_cols = edf.drop(columns=['y', 'y_change', 'round', 'round_change', 'idx']).columns

            # Create a dataframe with the mask values
            player_mask_df = pd.DataFrame(
                data=mask, 
                columns=mask_cols
            )

            for player in ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'T5', 'T6', 'T7', 'T8', 'T9']:
                
                # Get the index when the player died
                player_died_index = edf.loc[edf[f'{player}' + '_is_alive_change'] == -1].index

                # If the player didn't die
                if len(player_died_index) == 0:
                    continue
                
                # If the player died, add a 2 second buffer
                if len(player_died_index) > 0:
                    player_died_index = player_died_index[0] + 8


                # If the player died
                if player_died_index < len(player_mask_df):
                    player_columns = [col for col in player_mask_df.columns if player in col]
                    player_mask_df.loc[player_died_index:, player_columns] = 0

            masked_shap_values.append(shap_values[idx] * player_mask_df.values)

        return masked_shap_values

    def _SHAP_aggregate_shap_values(self, masked_shap_values, explainers, agg='mean'):

        # Max row number
        max_rows = max(shap_table.shape[0] for shap_table in masked_shap_values)

        # Initialize final array with zeros
        agg_shap_values = np.zeros((max_rows, masked_shap_values[0].shape[1]))

        # Sorok átlagolása a szabályaid szerint
        for i in range(1, max_rows + 1):  # Visszafelé számoljuk a sorokat
            rows_to_average = []
            for shap_table in masked_shap_values:
                if shap_table.shape[0] >= i:  # Ha az adott tömbben van i-edik (utolsó előtti, utolsó stb.) sor
                    rows_to_average.append(shap_table[-i])  # Hozzáadjuk az i-edik sorát
            # SUM
            if agg == 'sum':
                agg_shap_values[-i] = np.sum(rows_to_average, axis=0)
            # MEAN
            elif agg == 'mean':
                agg_shap_values[-i] = np.mean(rows_to_average, axis=0)


        explainer_expected_value = np.mean([explainer.expected_value for explainer in explainers])

        return agg_shap_values, explainer_expected_value



    def _SHAP_EXT_process_event_datasets(self):
        
        if self.edf_1 is None or self.edf_4 is None or self.edf_8 is None or self.edf_12 is None or self.edf_16 is None or self.edf_20 is None:
            for frame in self.previous_frames:
                exec(f'self.edf_{frame} = SnapshotEvents().get_round_events(self.graphs, self.predictions, {self.round_number}, shift_rate={frame}, dictionary=self.normalizing_dictionary)')



    def _SHAP_EVT_predict_proba(self, graphs, model, round_number: int):
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


       # Check if the dataset is temporal
        IS_TEMPORAL = False

        if graphs is not None and type(graphs[0])!=HeteroData:
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

        return predictions


    # The old round analyzer function, can analyze non-temporal rounds as well
    def _OLD_analyze_round(self, graphs, model, round_number: int, style: str = 'dark', model_code: str = None, fig_size=(20, 5), plt_title=None, plt_legend=True, plt_show=False, save_path: str = None, return_predictions: bool = False) -> None:
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
