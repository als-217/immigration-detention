import polars as pl

def harmonize_dtypes(df1, df2):
    for col in set(df1.columns) & set(df2.columns):
        t1, t2 = df1[col].dtype, df2[col].dtype
        if t1 == t2:
            continue
            
        # Determine target type
        if pl.Date in (t1, t2) or pl.Datetime in (t1, t2):
            target = t1 if t1 in (pl.Date, pl.Datetime) else t2
        elif pl.Utf8 in (t1, t2):
            target = pl.Utf8
        else:
            continue
        
        # Cast whichever doesn't match target
        if t1 != target:
            df1 = df1.with_columns(pl.col(col).cast(target, strict=False))
            print(col)
        if t2 != target:
            df2 = df2.with_columns(pl.col(col).cast(target, strict=False))
            print(col)

    return df1, df2