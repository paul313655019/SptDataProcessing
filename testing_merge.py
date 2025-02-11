# %%
import pandas as pd


# Create a sample dataframe
data1 = pd.DataFrame({
    'One': [1, 2, 3, 4, 6, 7, 8, 9],
    'Two': [11, 12, 13, 14, 16, 17, 18, 19],
    'Three': [111, 112, 113, 114, 116, 117, 118, 119],
    'ID': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
})

data2 = pd.DataFrame({
    'Four': [1, 2, 6, 7,],
    'Five': [11, 12, 16, 17],
    'Six': [111, 112, 116, 117],
    'ID': ['A', 'A', 'B', 'B',]
})

#%%
data1

#%%
data2
# %%
# data1['RowNumber'] = data1.groupby('ID').cumcount()
# data2['RowNumber'] = data2.groupby('ID').cumcount()

# data1 = pd.merge(data1, data2, on=['ID', 'RowNumber'], how='outer').drop('RowNumber', axis=1)

# data1 

def test_merge(df):
    data2 = pd.DataFrame({
    'Four': [1, 2, 6, 7,],
    'Five': [11, 12, 16, 17],
    'Six': [111, 112, 116, 117],
    'ID': ['A', 'A', 'B', 'B',]
    })
    df = df.reset_index(drop=True)
    df['Four'] = data2.loc[data2['ID'] == df['ID'].iloc[0], ['Four']].reset_index(drop=True)
    return df

data1 = data1.groupby('ID').apply(test_merge).reset_index(drop=True)
data1

# %%
