import os
import requests
from colorama import Fore, Style

dataset_urls = {
    "IMDb movie review dataset": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "Twitter sentiment analysis dataset": "https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2",
}

if not os.path.exists("dataset_csv"):
    os.makedirs("dataset_csv")

total_datasets = len(dataset_urls)
downloaded_datasets = 0

for dataset_name, dataset_url in dataset_urls.items():
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        file_name = "{}.csv".format(dataset_name)
        file_path = os.path.join("dataset_csv", file_name)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        downloaded_datasets += 1
        print(
            Fore.GREEN
            + "{} downloaded and saved as {}".format(dataset_name, file_path)
            + Style.RESET_ALL
        )
    except Exception as e:
        print(
            Fore.RED
            + "Failed to download {}. Error: {}".format(dataset_name, str(e))
            + Style.RESET_ALL
        )
    finally:
        progress = (downloaded_datasets / total_datasets) * 100
        print("Progress: {:.2f}%".format(progress))

print("All datasets downloaded.")


# Sentiment Analysis:
# IMDb movie review dataset: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# Twitter sentiment analysis dataset: https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2
# Amazon product review dataset: http://snap.stanford.edu/data/web-Amazon.html
# Large Movie Review Dataset (IMDb): https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# Sentiment140 dataset (Twitter): http://help.sentiment140.com/for-students/

# NLP:
# Cornell Movie Dialogs Corpus: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
# Amazon product review dataset: https://registry.opendata.aws/amazon-reviews/
# Twitter sentiment analysis dataset: https://www.kaggle.com/datasets/kazanova/sentiment140/download?datasetVersionNumber=2
# SST-5 Sentiment Analysis Dataset: https://nlp.stanford.edu/sentiment/index.html
# Yelp reviews dataset: https://www.kaggle.com/yelp-dataset/yelp-dataset
# Amazon Fine Food Reviews dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews

# Image Classification:
# CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
# ImageNet dataset: http://www.image-net.org/
# MNIST dataset: http://yann.lecun.com/exdb/mnist/
# Fashion-MNIST dataset: http://yann.lecun.com/exdb/mnist/
# Pascal VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/
# COCO (Common Objects in Context) dataset: http://cocodataset.org/
# SUN397 dataset: http://groups.csail.mit.edu/vision/SUN/
# Caltech-101 dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
# Caltech-256 dataset: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
