# MIT Sloan 2025

This document collects the relevant information/files about this project/repository for the MIT Sloan research paper competition. This includes the data parsing files and the GNN training files with some result plots. Although this repository holds most of the project, the data files are too big for a git upload; thus, links will be provided on which the used datasets are available.

## 1. Abstract

This section concludes the main, most relevant parts of this repository for the abstract phase of the MIT Sloan 2025 research paper competition.

### 1.1 Data

As mentioned in the abstract, a data transformation process is introduced in the paper capable of creating heterogeneous snapshots from the publicly available match replay files. These replay files are available for free download on a website called [HLTV](https://www.hltv.org/). These replay files are exact copies of the matches played, thus they can be replayed in game as well as parsed to get tracking level data. An example page for a match (with match download available under the **Rewatch** title) can be seen [here](https://www.hltv.org/matches/2375777/g2-vs-natus-vincere-blast-premier-fall-final-2024). After unzipping the file, the *.dem* files, the match replay files, can be used to extract tracking data.

The parse process is done using the *./package* folder of the root project. This package colects all the necessary functionalities related to parsing matches, collected to classes. The *graph* folder holds the classes responsible for creating the heterogeneous graph datasets available below. In the *preprocess* folder, different data imputation and normalization functions/classes are located, essential for the data transformation process.

The ***training dataset*** of the GNN (13 GB) is available on [this link](https://drive.google.com/drive/folders/1KeuvyCsqKCSFk6k9k1jQiHVtNR0kK5sq?usp=drive_link) (Google Drive).

The ***validation dataset*** (4 GB) is stored in [this folder](https://drive.google.com/drive/folders/1BQnRP2xySBx3kPGt0nLLJmfhODJGl6Tr?usp=drive_link) (Google Drive).

