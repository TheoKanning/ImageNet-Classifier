import os.path
import io
import urllib2
from httplib import HTTPException
from PIL import Image
from resizeimage import resizeimage

# Minimum size will eliminate single pixel and flickr missing photo images
MINIMUM_FILE_SIZE = 5000

IMAGENET_LINKS_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
IMAGE_DIRECTORY = "ImageNet_data/"

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


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
        resized.save(download_path, resized.format)
    except (IOError, HTTPException) as e:
        print e
        return False

    # Check if photo meets minimum size requirement
    size = os.path.getsize(download_path)
    if size < MINIMUM_FILE_SIZE:
        os.remove(download_path)
        print "Invalid Image: " + url
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

    for url in urllib2.urlopen(links_url):
        if images >= num_images:
            break
        image_name = url.rsplit('/')[-1]
        download_path = os.path.join(class_folder_path, image_name)

        if os.path.isfile(download_path):
            continue

        if download_image(url, download_path):
            images += 1
            print images

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


def main():
    download_class_images("n02084071", 15, IMAGE_DIRECTORY)


if __name__ == "__main__":
    main()
