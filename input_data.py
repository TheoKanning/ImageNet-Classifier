import os.path
import io
import urllib2
from httplib import HTTPException
from ssl import CertificateError
from PIL import Image
from resizeimage import resizeimage
import numpy as np

# Minimum size will eliminate single pixel and flickr missing photo images
MINIMUM_FILE_SIZE = 5000

IMAGENET_LINKS_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
IMAGE_DIRECTORY = "ImageNet_data/"
URL_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "bad_urls")

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


def is_good_url(url, class_id):
    """
    Checks if the given url is not on the class's bad url list
    :param url: url to be checked
    :param class_id: ImageNet class is, used to find appropriate list of bad urls
    :return: True if not on bad url list, False otherwise
    """

    if not os.path.exists(URL_DIRECTORY):
        return True

    file_name = class_id + ".txt"
    file_path = os.path.join(URL_DIRECTORY, file_name)

    if not os.path.exists(file_path):
        return True

    if url in open(file_path).read():
        return False

    return True


def store_bad_url(url, class_id):
    """
    Stores the given url in the class's bad url file
    :param url: url to be blacklisted
    :param class_id: class that url corresponds to
    :return:
    """

    if not os.path.exists(URL_DIRECTORY):
        os.mkdir(URL_DIRECTORY)

    file_name = class_id + ".txt"
    file_path = os.path.join(URL_DIRECTORY, file_name)

    with open(file_path, "a") as urls_file:
        urls_file.write(url)


def download_image(url, download_path):
    """
    Downloads a single image from a url to a specific path
    :param url: url of image
    :param download_path: full path of saved image file
    :return: true if successfully downloaded, false otherwise
    """

    print "Downloading from " + url

    try:
        fd = urllib2.urlopen(url, timeout=3)
        image_file = io.BytesIO(fd.read())
        image = Image.open(image_file)

        size = image.size
        if size[0] < IMAGE_WIDTH or size[1] < IMAGE_HEIGHT:  # Image too small
            return False

        resized = resizeimage.resize_cover(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        resized.save(download_path, 'jpeg', icc_profile=resized.info.get('icc_profile'))
    except (IOError, HTTPException, CertificateError) as e:
        print e
        return False

    # Check if photo meets minimum size requirement
    size = os.path.getsize(download_path)
    if size < MINIMUM_FILE_SIZE:
        os.remove(download_path)
        print "Invalid Image: " + url
        return False

    # Try opening as array to see if there are any errors
    try:
        load_image_as_array(download_path)
    except ValueError as e:
        os.remove(download_path)
        return False

    return True


def download_class_images(class_id, num_images, work_directory):
    """
        Downloads images of the corresponding class and puts them in a folder
    :param class_id: ImageNet id of the class, name of folder
    :param num_images: Maximum number of images to download
    :param work_directory: Directory where all image class folders are kept
    """
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)

    class_folder_path = os.path.join(work_directory, class_id)
    if not os.path.exists(class_folder_path):
        os.mkdir(class_folder_path)

    links_url = IMAGENET_LINKS_URL + class_id

    previous_images = os.listdir(class_folder_path)
    images = len(previous_images)
    print "{0} images found for class {1}".format(images, class_id)
    if images >= num_images:
        return

    for url in urllib2.urlopen(links_url):
        if images >= num_images:
            break
        url = url[:url.find('?')]  # remove all query strings
        if not is_good_url(url, class_id):
            continue

        image_name = url.rsplit('/')[-1]
        image_name = image_name.strip('\n\r')
        download_path = os.path.join(class_folder_path, image_name)

        if ".gif" in image_name:
            continue

        if os.path.isfile(download_path):
            continue

        if download_image(url, download_path):
            images += 1
            print images
        else:
            store_bad_url(url, class_id)
    print "{0} total images for {1}".format(images, class_id)


def download_dataset(class_ids, num_images):
    """
    Downloads and resizes images from the specified class ids and stores them in the work directory
    :param class_ids: list of ImageNet ids
    :param num_images: maximum number of images to download in each set
    """
    for class_id in class_ids:
        print "Starting download for " + class_id
        download_class_images(class_id, num_images, IMAGE_DIRECTORY)


def load_image_as_array(filepath):
    """
    Loads a single image and returns it as an array
    :param filepath: path to image file
    :return: array of image with size IMAGE_WIDTH*IMAGE_HEIGHT*3
    """
    im = Image.open(filepath)
    if len(np.shape(im)) is 2:
        array = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        array[:, :, :] = np.array(im)[:, :, np.newaxis]
        return array
    else:
        array = np.array(im)

    return array.astype(np.float32)


def create_one_hot_vector(index, length):
    """
    Creates a one-hot vector with that specified length and a 1 at the specified index
    :param index: index of 1 in vector
    :param length: length of vector
    :return: one-hot vector
    """
    assert length > 0, "One-hot vector length must be a positive number"
    assert 0 <= index < length, "Index (%s) must be between 0 and length(%s)" % (index, length)

    vector = np.zeros(length)
    vector[index] = 1
    return vector


def load_all_images(class_ids, num_images):
    """
    Loads images from the given classes and returns them in an array, along with a list of one-hot vector labels
    :param class_ids: ImageNet ids of classes to be retrieved
    :param num_images: maximum number of images to return per class, actual number may be smaller
    :return: list of images for each class, list of labels
    """

    download_dataset(class_ids, num_images)

    num_classes = len(class_ids)
    all_images = []
    all_labels = []

    for index, class_id in enumerate(class_ids):
        class_path = os.path.join(IMAGE_DIRECTORY, class_id)
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        num_class_files = min(len(files), num_images)
        for n in range(0, num_class_files):
            image = load_image_as_array(os.path.join(class_path, files[n]))
            all_images.append(image)
            all_labels.append(create_one_hot_vector(index, num_classes))

    return np.array(all_images), np.array(all_labels)


class DataSet(object):
    def __init__(self, images, labels):
        """Construct a DataSet using the given images and labels
        """

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns, 3] (assuming depth == 3)
        assert images.shape[3] == 3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2], 3)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def create_datasets(class_ids, num_samples=1000, val_fraction=0.1, test_fraction=0.1):
    """
    Creates training, validation, and test datasets from the given class ids using the desired proportions
    :param class_ids: ImageNet class ids of all classes to include
    :param num_samples: maximum sample images for each class
    :param val_fraction: fraction of images to put into validation set
    :param test_fraction: fraction of images to put into test set
    :return: training_set, validation_set, test_dataset
    """

    assert 0 <= val_fraction <= 0.25, "Validation fraction %s must be between 0 and 0.25" % val_fraction
    assert 0 <= test_fraction <= 0.25, "Test fraction %s must be between 0 and 0.25" % test_fraction

    all_images, all_labels = load_all_images(class_ids, num_samples)

    total_num_images = len(all_images)
    # Shuffle all images before splitting
    perm = np.arange(total_num_images)
    np.random.shuffle(perm)
    all_images = all_images[perm]
    all_labels = all_labels[perm]

    validation_size = int(total_num_images * val_fraction)
    test_size = int(total_num_images * test_fraction)

    validation_images = all_images[:validation_size]
    validation_labels = all_labels[:validation_size]

    test_images = all_images[validation_size:validation_size + test_size]
    test_labels = all_labels[validation_size:validation_size + test_size]

    train_images = all_images[validation_size + test_size:]
    train_labels = all_labels[validation_size + test_size:]

    # Mean normalization
    training_mean = np.mean(train_images)
    train_images -= training_mean
    validation_images -= training_mean
    test_images -= training_mean

    # Std dev normalization
    training_std_dev = np.std(train_images)
    train_images /= training_std_dev
    validation_images /= training_std_dev
    test_images /= training_std_dev

    train_dataset = DataSet(train_images, train_labels)
    validation_dataset = DataSet(validation_images, validation_labels)
    test_dataset = DataSet(test_images, test_labels)

    return train_dataset, validation_dataset, test_dataset


def main():
    create_datasets(["n02084071", "n01503061"], 15)


if __name__ == "__main__":
    main()
