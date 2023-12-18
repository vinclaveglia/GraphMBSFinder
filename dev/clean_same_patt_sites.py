import pandas as pd

df = pd.read_csv('Pattern_sites.csv')
print(df['same_patt_sites'])

def f(row):
    l = row.split(',')
    l2 = [x.split('|')[0] for x in l]
    return ','.join(l2)


df['same_patt_sites'] = df['same_patt_sites'].apply(f)


df.to_csv('Patt_sites4Milana.csv', index=False)



