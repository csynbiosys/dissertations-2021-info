import pandas as pd
from tabloo import show

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

df = pd.read_csv('doctorate.csv')

df = splitDataFrameList(df, 'Abbreviations', ', ')
df = splitDataFrameList(df, 'Abbreviations', 'or ')
df = splitDataFrameList(df, 'Abbreviations', 'Or ')

df = df[df['Abbreviations'] != ' ']

df['Abbreviations'] = df['Abbreviations'].str.replace('.', "").str.lower()
df['Abbreviations'] = df['Abbreviations'].str.replace(' ', "")

df = df.drop_duplicates(subset=['Degree','Abbreviations'])

df.to_csv('doctorate_abv2.csv',index=False)