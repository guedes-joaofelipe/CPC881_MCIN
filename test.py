import numpy    as np
import pandas   as pd
from utils      import load_data

import dirs

path1 = dirs.resultsPath+"ES_func5_dim2_succ_0.00"

data = load_data(path1)
# data = data.min(axis=1)
# print(data)

grouped = data.groupby(by="Run")

for group in grouped:
    print(group.size)
    # df = pd.DataFrame(group)
    # print(df.min(axis=1))

# for method in dir(grouped):
#     print(method)

# print(grouped)
