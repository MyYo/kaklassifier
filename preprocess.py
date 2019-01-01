from datetime import datetime
import numpy as np
import os
import pandas as pd
import re


def load_data_set():
    data_path = os.path.expanduser('Data/Data1.csv')
    df = pd.read_csv(data_path, comment='#')
    df.rename(columns={
        'Timestamp': 'timestamp',
        'Royal Canin Lowfat Can (13.6oz)': 'can',
        'Sweet Potato (11.5oz)': 'sweet',
        'Golden Potato (5.2oz)': 'golden',
        'Outcome': 'outcome',
        'Were We Sleeping / Jake Was Alone Just Before': 'alone',
        'Was it an Accident?': 'accident',
        'Notes': 'notes',
        'Poop Event': 'poop',
        'Feed Event': 'feed'
    }, inplace=True)
    return df


def accumulate_pastdata(df, hours_ago=24, span=3):
    df = df.sort_values(by='timestamp')
    for index, row in df.iterrows():
        if not row.poop:
            continue
        dt = datetime.fromtimestamp(row.timestamp)
        df.loc[index, 'hour_of_day'] = dt.hour
        for h in range(span, hours_ago + span, span):
            start = row.timestamp - (h * 3600)
            end = row.timestamp - (h - span) * 3600
            ddf = df[(df.timestamp > start) & (df.timestamp < end)]
            s = ddf.sum()
            start_str = h - span
            end_str = h
            df.loc[index, 'golden_{}_to_{}'.format(
                start_str, end_str)] = s.golden
            df.loc[index, 'can_{}_to_{}'.format(
                start_str, end_str)] = s.can
            df.loc[index, 'sweet_{}_to_{}'.format(
                start_str, end_str)] = s.sweet
            df.loc[index, 'outcome_{}_to_{}'.format(
                start_str, end_str)] = s.outcome
    return df


def to_timestamp(t):
        dt = datetime.strptime(t, '%m/%d/%Y %H:%M:%S')
        return int(dt.timestamp())


def pre_process_data_set(df):
    df = df.copy()
    df.timestamp = [to_timestamp(t) for t in df.timestamp.values]

    renames = {'Big One': 3, 'Regular': 2, 'No Poop': 0}
    for k, v in renames.items():
        df.loc[df.outcome == k, 'outcome'] = v

    df.loc[df.index, 'poop'] = False
    df.loc[df.index, 'feed'] = True
    df.loc[df.outcome >= 0, 'poop'] = True
    df.loc[df.outcome >= 0, 'feed'] = False
    df.loc[df.alone == 'Yes', 'alone'] = True
    df.loc[df.accident == 'Yes', 'accident'] = True
    df = accumulate_pastdata(df)
    return df


def load_data():
    df = load_data_set()
    df = pre_process_data_set(df)
    feature_columns = np.array([c for c in df.columns
                                if re.search('_to_', c)] + ['hour_of_day'])
    X = df.loc[df.poop, feature_columns].values
    y = (df.outcome[df.poop].values > 0).astype(np.int)
    return X, y, feature_columns
