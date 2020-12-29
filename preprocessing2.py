import DataReader 
import pandas as pd
import datetime as d


def main():
    r = DataReader.PdData()
    years = range(1,3)
    # train = dict(zip(years, 
    #                 [r.massaged_data(x, hourly=True).dropna() for x in years]))
    train = r.massaged_data(1, hourly=True)
    print("~"*25)
    print(train)
    # train_all = pd.concat(train.values(), axis=0).sort_index()
    print("~"*25)

    print(train.shape)
    
    timestamp = d.datetime.now().strftime("%d-%m-%Y(%H:%M:%S)")
    train.to_pickle(f"data/train_{timestamp}.p")


main()