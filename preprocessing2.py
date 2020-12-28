import DataReader 
import pandas as pd
import datetime as d


def main():
    r = DataReader.PdData()
    years = range(1,3)
    train = dict(zip(years, 
                    [r.massaged_data(x, hourly=True).dropna() for x in years]))
    train_all = pd.concat(train.values(), axis=0).sort_index()

    timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    train_all.to_pickle(f"data/train_{timestamp}.p")


main()