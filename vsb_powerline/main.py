import pandas as pd

# import argparse
from data_preprocessing import DataProcessor

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    dp = DataProcessor(150, 800000)
    df = pd.read_csv('data/raw_train.csv', compression='gzip')
    dp.transform(df)
