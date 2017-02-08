import csv
import numpy as np
from os import path


def import_data():

    data_file = "../GermainData.npy"
    data = []
    if not path.exists(data_file):

        print("Loading data from csv file...")

        headers = []

        with open('../lts_merged_2016.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')

            i = 0
            for row in reader:
                if i == 0:
                    headers = row

                else:
                    row_dic = {}
                    for key, content in zip(headers, row):

                        row_dic[key] = float(content.replace(",", "."))
                    data.append(row_dic)
                i += 1

        clean_data = []

        Session, realNumber = 1, 1

        d = {
            "own_good": [],
            "partner_type": [],
            "partner_good": [],
            "own_choice": [],
            "partner_choice": [],
            "prop": [],
            "reward": [],
            "cost": []
        }

        for row_idx in range(len(data)):

            if data[row_idx]["Session"] == Session and \
                    data[row_idx]["realNumber"] == realNumber:
                pass
            else:
                clean_data.append(d.copy())

                Session = data[row_idx]["Session"]
                realNumber = data[row_idx]["realNumber"]

                for key in d.keys():
                    d[key] = []

            d["own_good"].append(
                data[row_idx]["startGood"]
            )
            d["partner_type"].append(
                data[row_idx]["partnersType"]
            )
            d["partner_good"].append(
                data[row_idx]["proposedGood"]
            )
            d["own_choice"].append(
                data[row_idx]["willToExchange"]
            )
            d["partner_choice"].append(
                data[row_idx]["partnersWillToExchange"]
            )
            d["prop"].append(
                [
                    data[row_idx]["prop_pCyan_gYellow"],
                    data[row_idx]["prop_pYellow_gMagenta"],
                    data[row_idx]["prop_pMagenta_gCyan"]
                ]
            )
            d["reward"].append(
                data[row_idx]["currentConsumption"]
            )
            d["cost"].append(
                data[row_idx]["currentCost"]
            )

        np.save(data_file, clean_data)

    else:

        print("Loading data from npy file...")

        clean_data = np.load(data_file)

    return clean_data


def main():

    data = import_data()

    print("N sujets", len(data))


if __name__ == "__main__":

    main()