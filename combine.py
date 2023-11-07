import pandas as pd


def combine():
    data1 = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/dnn_unknown.csv')
    data2 = pd.read_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/ae_unknown.csv')

    a = data1.iloc[:, 1]
    b = data2.iloc[:, 0]

    a = pd.Series(a)
    b = pd.Series(b)

    c = pd.concat([a, b], axis=0)
    c = pd.DataFrame(c)
    c.columns = ['col1']
    c = c.sort_values(by=['col1'])
    c = c.drop_duplicates()
    c.reset_index(inplace=True, drop=False)
    c.drop(c.columns[[0]], axis=1, inplace=True)
    c.to_csv('C:/Users/user/Downloads/NSL_GUI-20221220T055912Z-001/NSL_GUI/unknown.csv')
    return (len(c))


combine()
