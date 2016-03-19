import input_data
import numpy as np

NUM_IMAGES = 100

classes = np.array([["dog", "n02084071"],
                    ["cat", "n02121808"]])


def main():
    ids = classes[:, 1]
    input_data.download_dataset(ids, NUM_IMAGES)


if __name__ == "__main__":
    main()
