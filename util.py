import csv
import numpy as np
import kagglehub as kh


def load_data(path: str, y_column=0, skip_first_column=False, add_intercept=True):
    with open(path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)  # skip header
        data = np.array(list(csvreader))
        if skip_first_column:
            data = data[:, 1:]
        m = np.shape(data)[0]
        y = data[:, y_column]
        x = np.delete(data, y_column, axis=1)
        if add_intercept:
            x = np.concatenate((np.ones((m, 1)), x), axis=1)
        return x, y


def load_from_kaggle(url: str, filename: str, y_column=0, skip_first_column=False, add_intercept=True):
    path = kh.dataset_download(url)
    return load_data(f'{path}/{filename}', y_column, skip_first_column, add_intercept)
