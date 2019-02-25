import pandas as pd
from data_preprocessing import DataProcessor

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    peak_threshold = 15
    dp = DataProcessor(150, 800000, peak_threshold)
    df = pd.read_csv('data/raw_train.csv', compression='gzip')
    output_df = dp.transform(df)
    print(output_df.head())
    output_df.to_csv('data/transformed_train.csv', compression='gzip')
