import csv
import numpy as np
from os import path


def read_from_csv(csv_file):

    data = []
    headers = []

    with open(csv_file, 'r') as csv_content:
        reader = csv.reader(csv_content, delimiter=';')

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

    return data


def format_data(csv_data):

    clean_data = []

    Session, realNumber = 1, 1

    labels = [
        "subject_good",
        "subject_good",
        "partner_good",
        "subject_choice",
        "partner_choice",
        "partner_type",
        "prop",
        "u",
        "storing_costs"
    ]

    # Data for a single subject
    d = {}
    for label in labels:
        d[label] = []

    for row_idx in range(len(csv_data)):

        if csv_data[row_idx]["Session"] == Session and \
                        csv_data[row_idx]["realNumber"] == realNumber:
            pass

        else:

            d["prop"] = np.asarray(d["prop"])
            d["u"] = max(d["u"])
            d["beta"] = 0.9

            c = sorted(np.unique(d["storing_costs"]))
            if len(c) < 3:
                if 4 in c:
                    c = [1, 4, 9]
                else:
                    c = [1, 3, 9]

            assert c[0] < c[1] < c[2], c
            # Keep only the 3 storing costs and not the history of the costs
            d["storing_costs"] = c

            clean_data.append(d.copy())

            Session = csv_data[row_idx]["Session"]
            realNumber = csv_data[row_idx]["realNumber"]

            for key in d.keys():
                d[key] = []

        d["subject_good"].append(
            int(csv_data[row_idx]["startGood"] - 1)
        )
        d["partner_type"].append(
            int(csv_data[row_idx]["partnersType"] - 1)
        )
        d["partner_good"].append(
            int(csv_data[row_idx]["proposedGood"] - 1)
        )
        d["subject_choice"].append(
            int(csv_data[row_idx]["willToExchange"])
        )
        d["partner_choice"].append(
            int(csv_data[row_idx]["partnersWillToExchange"])
        )
        d["prop"].append(
            [
                csv_data[row_idx]["prop_pCyan_gYellow"],
                csv_data[row_idx]["prop_pYellow_gMagenta"],
                csv_data[row_idx]["prop_pMagenta_gCyan"]
            ]
        )
        d["u"].append(
            int(csv_data[row_idx]["currentConsumption"])
        )
        d["storing_costs"].append(
            int(csv_data[row_idx]["currentCost"])
        )

    return clean_data


def import_from_csv_file(csv_file):

    csv_data = read_from_csv(csv_file)

    return format_data(csv_data)


def import_data(force=False):

    npy_file = "../GermainData.npy"
    csv_file = "../lts_merged_2016.csv"

    if not path.exists(npy_file) or force:

        print("Loading data from CSV file...")
        data = import_from_csv_file(csv_file=csv_file)
        np.save(npy_file, data)

    else:

        print("Loading data from NPY file...")
        data = np.load(npy_file)

    print("Data loaded.")
    print()

    return data


def main():

    data = import_data(force=True)

    print("N sujets", len(data))


if __name__ == "__main__":

    main()