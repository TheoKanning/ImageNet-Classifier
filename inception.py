import input_data
import numpy as np

NUM_IMAGES = 100

classes = np.array([["dog", "n02084071"],
                    ["cat", "n02121808"],
                    ["bird", "n01503061"],
                    ["orange", "n07747607"],
                    ["apple", "n07739125"],
                    ["keyboard", "n03614007"],
                    ["computer mouse", "n03793489"],
                    ["desk", "n03179701"],
                    ["monitor", "n03782006"],
                    ["book", "n02870526"],
                    ["pen", "n03906997"],
                    ["pencil", "n03908204"],
                    ["chair", "n03001627"],
                    ["sword", "n04373894"],
                    ["cup", "n03147509"],
                    ["shirt", "n04197391"],
                    ["shoe", "n04200000"],
                    ["car", "n02960352"],
                    ["door", "n03222176"]
                    ])


def main():
    ids = classes[:, 1]
    input_data.download_dataset(ids, NUM_IMAGES)




if __name__ == "__main__":
    main()
