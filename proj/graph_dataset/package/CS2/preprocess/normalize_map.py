from sklearn.preprocessing import MinMaxScaler
from joblib import dump


class NormalizeMap:


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Normalize map graph
    # --------------------------------------------------------------------------------------------

    # Normalize the map graph nodes dataset
    def normalize_map_graph(
        self, 
        nodes, 
        pos_col_names=['X', 'Y', 'Z'], 
        scaler_operation='none', 
        scaler_save_path=None
    ):
        """
        Normalize the map node dataset's X, Y and Z coordinates.
        
        Parameters:
            - nodes: The nodes dataset of the map.
            - pos_col_names: The names of the positional columns to normalize. Default is ['X', 'Y', 'Z'].
            - scaler_operation: The operation to perform with the scaler. It can be 'save', 'return' or 'none'. Default is 'none'.
            - scaler_save_path: The path to which the scaler should be saved. Useful only if the scaler_operation is 'save'. Default is None.
        """

        # Check whether the filename ends with '.pkl'
        if scaler_save_path != None and scaler_save_path[-4:] != '.pkl':
            scaler_save_path += '.pkl'

        # Fit the scaler and transform the nodes dataset
        map_graph_scaler = MinMaxScaler()
        map_graph_scaler.fit(nodes[pos_col_names])
        nodes[pos_col_names] = map_graph_scaler.transform(nodes[pos_col_names])
        
        # If the scaler operation is 'save', save the model and return the nodes dataset
        if scaler_operation == 'save':

            if scaler_save_path == None:
                print('Path or filename was not declared, unable to save the scaler.')
                return nodes
            
            dump(map_graph_scaler, scaler_save_path)
            return nodes

        # If the scaler operation is 'return', return both the nodes dataset and the scaler
        elif scaler_operation == 'return':
            return nodes, map_graph_scaler
        
        # If the scaler operation is 'none', return only the nodes dataset
        else:
            return nodes
    
