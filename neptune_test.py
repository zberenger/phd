import numpy as np
import neptune
from neptune.utils import stringify_unsupported
import torch

run = neptune.init_run(project="zoeb/dilation",
                       api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNzQ4OWYzOC1jYzIzLTQ1NjEtOTliNC1jZmRmYTlmZjE3M2QifQ==",
                       mode="offline")  # your credentials

tmp1 = np.random.rand(10)
tmplist = [0, 1, 2, 3, 4, 5]
run["results/tmplist"].extend(tmplist)
run["results/tmp1"].extend(tmp1.tolist())
tmp2 = np.ones((10, 2))
run["results/tmp2"].extend(tmp2.reshape(-1).tolist())
run["results/tmp2strunsup"] = stringify_unsupported(tmp2)
tmptorch = torch.ones((20))
run["results/tmptorch"].extend(tmptorch.tolist())
tmptorch2d = torch.zeros((5, 20))
run["results/tmptorch2d"].extend(tmptorch2d.tolist()) # fonctionne pas
tmparraytensor = np.asarray([torch.ones(3) for i in range(10)])
run["results/tmparraytensor"].extend(tmparraytensor.tolist()) # fonctionne pas
run["results/tmparraytensorstrunsup"] = stringify_unsupported(tmparraytensor)
run["results/tmptorch2dstrunsup"] = stringify_unsupported(tmptorch2d)
