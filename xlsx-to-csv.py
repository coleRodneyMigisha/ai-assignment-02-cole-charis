import pandas as pd


def createCSVs(xls):
    """
    :param xls: 
    :return: two CSVs from the sheets
    """
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        df.to_csv(f'{sheet_name}.csv', index=False)


xls = pd.ExcelFile('dsca.xlsx')
createCSVs(xls)
