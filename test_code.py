import numpy as np
import pandas as pd

# Sample DataFrame
data = {'A': [1, 2, None], 'B': [4, None, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# Find indices of cells that are not None
nonna_ind = df.notna().stack().index.codes
# produces two lists with row indices and column indices

indices = list()
values = []
for i in range(len(nonna_ind[0])):
    row = nonna_ind[0][i]
    col = nonna_ind[1][i]
    tmp_tuple = (row,col)
    indices.append(tmp_tuple)
    values.append(df.iloc[row,col])
    pass



import pandas as pd
import numpy as np

def randomly_select_indices(df, percentage):
    """
    Randomly select a percentage of cells from the DataFrame and return their indices and values.

    Parameters:
    df (pd.DataFrame): The DataFrame to select from.
    percentage (float): The percentage of cells to select.

    Returns:
    tuple: A tuple containing the selected indices, the corresponding values as a list, and a DataFrame with non-selected cells set to NA.
    """
    # Calculate the total number of cells
    total_cells = df.size
    
    # Calculate the number of cells to select
    num_samples = int(total_cells * percentage / 100)
    
    # Randomly select cell indices
    selected_indices = np.random.choice(total_cells, num_samples, replace=False)
    
    # Convert flat indices to row, col indices
    row_indices, col_indices = np.unravel_index(selected_indices, df.shape)
    
    # Get the values corresponding to the selected indices
    selected_values = [df.iat[row, col] for row, col in zip(row_indices, col_indices)]
    
    # Create a copy of the DataFrame and set non-selected cells to NA
    df_with_na = df.copy()
    mask = np.ones(df.shape, dtype=bool)
    mask[row_indices, col_indices] = False
    df_with_na[mask] = np.nan
    
    return list(zip(row_indices, col_indices)), selected_values, df_with_na

# Example usage
data = {'A': [1, 2, 3, 4, 5], 'B': [6, 7, 8, 9, 10], 'C': [11, 12, 13, 14, 15]}
df = pd.DataFrame(data)

# Set the percentage of data to select
percentage = 40

# Get the randomly selected indices and their values
indices, values, df_with_na = randomly_select_indices(df, percentage)

print("Selected Indices:", indices)
print("Selected Values:", values)
print("DataFrame with Non-Selected Cells Set to NA:\n", df_with_na)


pass


