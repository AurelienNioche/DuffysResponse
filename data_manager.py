import csv
import numpy as np
from os import path


def import_data():

    data_file = "../GermainData.npy"
    data = []
    if not path.exists(data_file):

        print("Loading data from CSV file...")

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
            "subject_good": [],
            "partner_type": [],
            "partner_good": [],
            "subject_choice": [],
            "partner_choice": [],
            "prop": [],
            "u": [],
            "c": []
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

            d["subject_good"].append(
                int(data[row_idx]["startGood"] - 1)
            )
            d["partner_type"].append(
                int(data[row_idx]["partnersType"] - 1)
            )
            d["partner_good"].append(
                int(data[row_idx]["proposedGood"] - 1)
            )
            d["subject_choice"].append(
                int(data[row_idx]["willToExchange"])
            )
            d["partner_choice"].append(
                int(data[row_idx]["partnersWillToExchange"])
            )
            d["prop"].append(
                [
                    data[row_idx]["prop_pCyan_gYellow"],
                    data[row_idx]["prop_pYellow_gMagenta"],
                    data[row_idx]["prop_pMagenta_gCyan"]
                ]
            )
            d["u"].append(
                int(data[row_idx]["currentConsumption"])
            )
            d["c"].append(
                int(data[row_idx]["currentCost"])
            )

        np.save(data_file, clean_data)

    else:

        print("Loading data from NPY file...")

        clean_data = np.load(data_file)

    print("Data loaded.")

    return clean_data


def main():

    data = import_data()

    print("N sujets", len(data))


if __name__ == "__main__":

    main()