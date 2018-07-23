import numpy    as np
import pandas   as pd
from glob       import glob
from tqdm       import tqdm

from utils      import load_data
import dirs

targetError = 1e-8
numRuns     = 50

dim         = 10
algorithm = "ES"

## Table1: Error statistics per function
# folder = "./resultsaux/"
folder = dirs.resultsPath+algorithm+"/"

errorTable = pd.DataFrame()
for file in tqdm(glob(folder+"*dim"+str(dim)+"*")):
    file = file.replace("\\", "/")
    data = load_data(file)

    # Append to table best error of each Run
    subTable = data.groupby(by="Run").min().T
    errorTable = errorTable.append(subTable)

# Count sucessess as error <= targetError
successTable = np.sum(np.where(errorTable <= targetError, 1, 0), axis=1)/errorTable.shape[1]

# Compose statistics table
table1 = pd.DataFrame(data={'Best': errorTable.min(axis=1), 'Worst':errorTable.max(axis=1),
                            'Median':errorTable.median(axis=1), 'Mean':errorTable.mean(axis=1),
                            "Std": errorTable.std(axis=1), "Success Rate": successTable})

# Save as excel file
savePath = dirs.tablesPath+"{}_table1_dim{}.xlsx".format(algorithm, dim)
table1.to_excel(savePath, float_format="%.6f", index_label="F#")

## Table2: Best error evolution per generation per function
for file in tqdm(glob(folder+"*")):
    file = file.replace("\\", "/")

    data = pd.read_hdf(file)
    errorTable = pd.DataFrame()

    fillVal = 1e15
    data = data.fillna(value=fillVal)

    key = "F"+file.split("_")[1][-2:]

    # For each run, get evolution of best errors per generation
    for run in range(0, numRuns):
        index = (data['Run'] == run)

        subTable = pd.DataFrame(data.values[index, :])
        generations = subTable.shape[0]

        # Only include a pre-determined set of generations
        fesIndex = (generations - 1)*np.array([0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5,
                                        0.6, 0.7, 0.8, 0.9, 1.0])
        fesIndex = fesIndex.round()

        # Get only the best individuals
        subTable = subTable.iloc[:, :-1].min(axis=1, skipna=True)

        # Append Run data to the table
        errorTable['Run {:2d}'.format(run)] = subTable.iloc[fesIndex.astype(int)]

    print("\n", key)
    print(errorTable)

    # Add a column with each function's mean error over all runs
    errorTable["Mean"] = errorTable.mean(axis=1, skipna=True)

    # Save as excel file
    savePath = dirs.tablesPath+"{}_table2_{}_dim{}.xlsx".format(algorithm, key, dim)
    errorTable.to_excel(savePath, float_format="%.8f", index_label='Gen')
