import csv
import numpy as np
from os import path

def import_data():
    data = []

    data_file = "GermainData.npy"
    headers_file = "GermainHeaders.npy"

    if not path.exists(data_file):

        print("Loading data from csv file...")

        headers = None

        with open('../lts_merged_2016.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')

            i = 0
            for row in reader:
                if i == 0:
                    headers = row

                else:
                    data.append([float(i.replace(",", ".")) for i in row])
                i += 1

        data = np.asarray(data)
        headers = np.asarray(headers)

        np.save(data_file, data)
        np.save(headers_file, headers)

    else:

        print("Loading data from npy file...")

        data = np.load(data_file)
        headers = np.load(headers_file)

    return data, headers


def main():

    data, headers = import_data()


if __name__ == "__main__":

    main()
