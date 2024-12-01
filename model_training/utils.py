import pandas as pd


def read_dataset(path, text_column="text", target_column="target"):
    df = pd.read_csv(path)
    df.dropna(inplace=True, axis=0)
    df.reset_index(inplace=True, drop=True)
    required_columns = ["ID", text_column, target_column]
    if 'ID' not in df.columns:
        df['ID'] = range(1, len(df) + 1)

    print(required_columns)
    df = df[required_columns]
    
    df = df.rename(columns={
        text_column: "text",
        target_column: "target"
    })

    return df
