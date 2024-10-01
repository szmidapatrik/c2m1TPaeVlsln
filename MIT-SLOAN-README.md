# MIT Sloan 2025 -  ***Evaluating Player Actions in Professional Counter Strike using Temporal Heterogeneous Graph Neural Networks***

*Patrik Peter Szmida*, *Laszlo Toka*

This document collects the relevant information/files about this project/repository for the MIT Sloan research paper competition. This includes the data parsing files, the GNN training files with some result plots, as well as the best models. Although this repository holds most of the project, the data files are too big for a git upload; thus, links will be provided on which the used datasets are available.

## 1. Abstract

This section concludes the main, most relevant parts of this repository for the abstract phase of the MIT Sloan 2025 research paper competition.

### 1.1 Data

As mentioned in the abstract, a data transformation process is introduced in the paper capable of creating heterogeneous snapshots from the publicly available match replay files. These replay files are available for free download on a website called [HLTV](https://www.hltv.org/). These replay files are exact copies of the matches played, thus they can be replayed in game as well as parsed to get tracking level data. An example page for a match (with match download available under the **Rewatch** title) can be seen [here](https://www.hltv.org/matches/2375777/g2-vs-natus-vincere-blast-premier-fall-final-2024). After unzipping the file, the *.dem* files, the match replay files, can be used to extract tracking data.

The parse process is done using the *./package* folder of the root project. This package colects all the necessary functionalities related to parsing CS2 matches, collected to classes. The *./package/CS2/graph* folder holds the classes responsible for creating the heterogeneous graph datasets available below. In the *CS2/preprocess* folder, different data imputation and normalization functions/classes are located, essential for the data transformation process.

The ***training dataset*** of the GNN (13.5 GB) is available on [this link](https://drive.google.com/drive/folders/1KeuvyCsqKCSFk6k9k1jQiHVtNR0kK5sq?usp=drive_link) (Google Drive).

The ***validation dataset*** (4.1 GB) is stored in [this folder](https://drive.google.com/drive/folders/1BQnRP2xySBx3kPGt0nLLJmfhODJGl6Tr?usp=drive_link) (Google Drive).

### 1.2 Models

The models used for prediction are stored under the *./models/gnn* folder, with the folder names indicating the date of training. The current best model, used for the player action evaluation plot in the abstract, is located in the *240930_5* folder, with all information related to the training saved. The training was done in the *./proj/graph_nn* folder, the jupyter notebooks contain the results of some of the latest trainings.

Current best model (used in the abstract): ***./model/gnn/240930_5/epoch_3.pt***.

### 1.3 Plots and visualization

The abstract included two figures about the project. The first, the example heterogeneous graph snapshot is created using the project's package. The visualization folder provides a class for plotting heterogeneous snapshots. The figure included in the abstract can be seen in the *./proj/graph_dataset/CS2_dataset_create.ipynb* file. The creation of the second plot can be seen at the end of *./proj/graph_nn/gnn_thesis2_large.ipynb* file, using the model and the dataset mentioned above.