# BUGFIXES

This documents hold information about some bugs found that are unrelated to the code content of this library (e.g. Pytorch error). These issues are listed below as well as the solutions (if there is one).

----------

## 1. PyG Temporal - Import error (2024. 09. 10.)

*Description:* An error occured when importing ```from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal```, and likely would occur at any other import.

*Error message*:
```
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[2], line 5
      3 import torch
      4 from torch_geometric.utils import to_dense_adj
----> 5 import torch_geometric_temporal
      7 import pandas as pd
      8 import polars as pl

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\torch_geometric_temporal\__init__.py:1
----> 1 from torch_geometric_temporal.nn import *
      2 from torch_geometric_temporal.dataset import *
      3 from torch_geometric_temporal.signal import *

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\torch_geometric_temporal\nn\__init__.py:2
      1 from .recurrent import *
----> 2 from .attention import *
      3 from .hetero import *

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\torch_geometric_temporal\nn\attention\__init__.py:6
      4 from .gman import GMAN, SpatioTemporalEmbedding, SpatioTemporalAttention
      5 from .mtgnn import MTGNN, MixProp, GraphConstructor
----> 6 from .tsagcn import GraphAAGCN, AAGCN
      7 from .dnntsp import DNNTSP

File ~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\torch_geometric_temporal\nn\attention\tsagcn.py:6
      4 import torch.nn as nn
      5 from torch.autograd import Variable
----> 6 from torch_geometric.utils.to_dense_adj import to_dense_adj
      7 import torch.nn.functional as F
     10 class GraphAAGCN:

ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'
```


*Reason*: The version of PyG is ```2.5.3```, but PyG temporal does not support PyG ```2.5.X```, only below.

*Solution*: The issue is caused because one function in PyG that is used in PyG Temporal was moved while PyG updated to ```2.5.X```. Fixing it consists of the following steps:

  1. Navigate to ```C:\Users\*username*\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\```, where the isntalled packages are located for the used kernel (note that ```Python.3.12_qbz5n2kfra8p0``` is the folder of the kernel, in other kernel it has a different name).

  2. In the packages fodler, navigate to ```torch_geometric_temporal\nn\attention```, the error is caused in this folder, by the ```tsagc.py``` file.

  3. Open the ```tsagc.py``` file.

  4. In the imports, make the following changes:

  Default version of the file:
  ```python
  ...
  from torch.autograd import Variable
  from torch_geometric.utils.to_dense_adj import to_dense_adj  # <----- change this line
  import torch.nn.functional as F
  ...
  ```

  After modification:
  ```python
  ...
  from torch.autograd import Variable
  from torch_geometric.utils import to_dense_adj  # <----- this is the correct version
  import torch.nn.functional as F
  ...

  ```
