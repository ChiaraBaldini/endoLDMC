"""
Calculates the Frechet Inception Distance between two distributions, using chosen feature extractor model.

RadImageNet Model source: https://github.com/BMEII-AI/RadImageNet
RadImageNet InceptionV3 weights (original, broken since 11.07.2023): https://drive.google.com/file/d/1p0q9AhG3rufIaaUE1jc2okpS8sdwN6PU
RadImageNet InceptionV3 weights (for medigan, updated link 11.07.2023): https://drive.google.com/drive/folders/1lGFiS8_a5y28l4f8zpc7fklwzPJC-gZv

Usage:
    python fid.py dir1 dir2 
"""
import argparse
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_probability as tfp
import wget
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm
from csv import writer
from datetime import datetime
from pathlib import Path
import glob

tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

from tensorflow.python.client import device_lib

random.seed(123)
np.random.seed(123)

img_size = 512  #model input size, e.g. 224 for InceptionV3, 512 for ResNet50
num_batches = 64
num_batches = 1
CUSTOMODEL_WEIGHTS = ["your_custom_model_weights.h5"]
IMAGENET_TFHUB_URL = "https://tfhub.dev/tensorflow/tfgan/eval/inception/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the Frechet Inception Distance between two distributions using RadImageNet model."
    )
    parser.add_argument(
        "dataset_path_1",
        type=str,
        help="Path to images from first dataset",
    )
    parser.add_argument(
        "dataset_path_2",
        type=str,
        help="Path to images from second dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="imagenet",
        help="Use RadImageNet feature extractor for FID calculation",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Describe the run e.g. state the checkpoint name and important config info",
    )
    parser.add_argument(
        "--lower_bound",
        action="store_true",
        help="Calculate lower bound of FID using the 50/50 split of images from dataset_path_1",
    )
    parser.add_argument(
        "--normalize_images",
        action="store_true",
        help="Normalize images from both data sources using min and max of each sample",
    )
    parser.add_argument(
        "--is_split_per_patient",
        action="store_true",
        default=False,
        help="If the dataset is split to calculate FID, then split per patient",
    )
    parser.add_argument(
        "--reverse_split_ds1",
        action="store_true",
        help="if the dataset from `dataset_path_1` is split in a deterministic way, reverse that splitting order",
    )
    parser.add_argument(
        "--reverse_split_ds2",
        action="store_true",
        help="if the dataset from `dataset_path_2` is split in a deterministic way, reverse that splitting order",
    )
    parser.add_argument(
        "--is_only_splitted_loaded",
        action="store_true",
        help="If the dataset is plit into two, then only the first split is returned and the second is ignored",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=9999999, #3000,  # 4999, #2999, #49,
        help="Max number of images to load from each data source",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default= "0001",
        help="The DCE-MRI postcontrast phase. By default this is phase 1",
    )
    parser.add_argument(
        "--secondphase",
        type=str,
        default= "0000",
        help="In case we don't want to compare to the precontrast images, but rather compare two different DCE-MRI phases. ",
    )
    parser.add_argument(
        "--segmentation_path",
        type=str,
        default= None,
        help="The path to where the segmentations are stored. If provided, the segmentations will be used to extract the bounding box region of interest from the images before computing metrics.",
    )
    args = parser.parse_args()
    return args


def split_per_patient(directory, reverse_split=False):
    set_of_patient_ids = set([filename[0:13] for filename in os.listdir(directory)])
    if reverse_split:
        list1 = [patient_id for count, patient_id in enumerate(set_of_patient_ids) if not count % 2 == 0]
    else:
        list1 = [patient_id for count, patient_id in enumerate(set_of_patient_ids) if count % 2 == 0]
    return list1


def load_images(directory, normalize=False, split=False, is_split_per_patient=True, reverse_split=False,
                is_only_splitted_loaded=False, limit=None):
    """
    Loads images from the given directory.
    If split is True, then half of the images is loaded to one array and the other half to another.
    """

    print(f"Loading images from {directory} ...")
    images = None
    subset_1 = None
    subset_2 = None
    patient_list_1 = []
    if split:
        subset_1 = []
        subset_2 = []
        if is_split_per_patient:
            patient_list_1 = split_per_patient(directory, reverse_split=reverse_split)
    else:
        images = []

    for count, filename in tqdm(enumerate(os.listdir(directory)), total=limit if limit else len(os.listdir(directory))):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            # img_array = np.array(img)
            # min_val = img_array.min()
            # max_val = img_array.max()
            # print(f"Min value: {min_val}")
            # print(f"Max value: {max_val}")
            if normalize:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                # img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
                # print(f"Min value: {np.array(img).min()}")
                # print(f"Max value: {np.array(img).max()}")
            if len(img.shape) > 2 and img.shape[2] == 4:
                img = img[:, :, :3]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            if split and not split_per_patient:
                if count % 2 == 0 or (not (count % 2 == 0) and reverse_split):
                    subset_1.append(img) if len(subset_1) < limit else None
                elif not is_only_splitted_loaded:
                    subset_2.append(img) if len(subset_2) < limit else None
            elif split and is_split_per_patient:
                if any(patient_id in filename for patient_id in patient_list_1):
                    # print(f"Added to subset 1: {filename}")
                    subset_1.append(img) if len(subset_1) < limit else None
                elif not is_only_splitted_loaded:
                    # print(f"Added to subset 2: {filename}")
                    subset_2.append(img) if len(subset_2) < limit else None
            else:
                images.append(img) if len(images) < limit else None
        if limit is not None and (
                (images is not None and len(images) == limit) or
                (subset_1 is not None and len(subset_1) == limit)):
            break
    if split:
        if is_only_splitted_loaded:
            return np.array(subset_1), None
        else:
            return np.array(subset_1), np.array(subset_2)
    else:
        return np.array(images), None

def load_images_from_filelist(file_names, normalize=False, split=False, limit=None):
    """
    Loads images from the given directory.
    If split is True, then half of the images is loaded to one array and the other half to another.
    """

    if split:
        subset_1 = []
        subset_2 = []
    else:
        images = []
    for count, filename in enumerate(file_names):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(filename)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            # img_array = np.array(img)
            # min_val = img_array.min()
            # max_val = img_array.max()
            # print(f"Min value: {min_val}")
            # print(f"Max value: {max_val}")
            if normalize==True:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                # img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
                # print(f"Min value 2: {np.array(img).min()}")
                # print(f"Max value 2: {np.array(img).max()}")
            if len(img.shape) > 2 and img.shape[2] == 4:    
                img = img[:, :, :3]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=2)
            if split:
                if count % 2:
                    subset_1.append(img)
                else:
                    subset_2.append(img)
                if limit is not None and count >= limit*2 and len(subset_1) >= limit and len(subset_2) >= limit:
                    break
            else:
                images.append(img)
                if limit is not None and count == limit:
                    break
    if split:
        return np.array(subset_1), np.array(subset_2)
    else:
        return np.array(images), None

def check_model_weights(model_name):
    """
    Checks if the model weights are available and download them if not.
    """
    model_weights_path = None
    if model_name == "custom":
        for weight_path in CUSTOMODEL_WEIGHTS:
            if os.path.exists(weight_path):
                model_weights_path = weight_path
                break
        return model_weights_path


def _custom_fn(images):
    """
    Get custom model, e.g. ResNet50
    """
    model_weights_path = None
    for weight_path in CUSTOMODEL_WEIGHTS:
        if os.path.exists(weight_path):
            model_weights_path = weight_path
            break
    model = ResNet50(
        weights=None,
        input_shape=(img_size, img_size, 3),
        include_top=False,
        pooling="avg",
    )
    model.load_weights(model_weights_path, by_name=True, skip_mismatch=True)
    output = model(images)
    output = tf.nest.map_structure(tf.keras.layers.Flatten(), output)
    return output


def get_classifier_fn(model_name="imagenet"):
    """
    Get model as TF function for optimized inference.
    """
    check_model_weights(model_name)

    if model_name == "custom":
        return _custom_fn
    elif model_name == "imagenet":
        return tfgan.eval.classifier_fn_from_tfhub(IMAGENET_TFHUB_URL, "pool_3", True)
    else:
        raise ValueError("Model {} not recognized".format(model_name))


def calculate_fid(
        directory_1,
        directory_2,
        model_name,
        phase,
        secondphase,
        lower_bound=False,
        normalize_images=False,
        is_split_per_patient=False,
        reverse_split_ds1=False,
        reverse_split_ds2=False,
        is_only_splitted_loaded=False,
        limit=None,
        enforce_src_target_file_correspondence=True,
        segmentation_path=None,
):
    """
    Calculates the Frechet Inception Distance between two distributions using chosen feature extractor model.
    """
    if limit is None:
        limit = min(len(os.listdir(directory_1)), len(os.listdir(directory_2)))

    if lower_bound and is_only_splitted_loaded:
        raise NotImplementedError(
            f"Defining the lower bound is not compatible with only returning part of the dataset (is_only_splitted_loaded={is_only_splitted_loaded}).")
    elif lower_bound and not is_only_splitted_loaded:
        images_1, images_2 = load_images(directory_1, split=True, limit=limit, normalize=normalize_images,
                                         is_split_per_patient=is_split_per_patient, reverse_split=reverse_split_ds1,
                                         is_only_splitted_loaded=is_only_splitted_loaded)
        if segmentation_path is not None:
            raise NotImplementedError(
                f"Segmentation is not compatible with lower bound calculation (lower_bound={lower_bound}). You specified a segmentation_masks mask directory: {segmentation_path}.")
    else:
        if enforce_src_target_file_correspondence:
            file_list_1, file_list_2 = check_if_files_correspond(directory_1=directory_1, directory_2=directory_2, rename=True, enforce_strict_file_correspondence=True, phase=phase, secondphase=secondphase)   #rename was True
            if segmentation_path is not None:
                print(f"Len of segmentation masks = {len(sorted(glob.glob(f'{segmentation_path}/*.png')))}")
                print(f"Len of images 1 = {len(file_list_1)}")
                file_list_1, segmentation_masks = check_if_files_correspond(directory_1=directory_1, directory_2=segmentation_path, rename=True, enforce_strict_file_correspondence=True, phase=phase,secondphase="0001")
        images_1, _ = load_images_from_filelist(file_list_1, split=is_split_per_patient, limit=limit-1, normalize=normalize_images)
        images_2, _ = load_images_from_filelist(file_list_2, split=is_split_per_patient, limit=limit-1, normalize=normalize_images)

        if segmentation_path is not None:
            print(f"Found {len(segmentation_masks)} files in {segmentation_path}. Limit={limit}, phase={phase}, secondphase={secondphase}")
            masks, _ = load_images_from_filelist(segmentation_masks, split=is_split_per_patient, limit=limit-1, normalize=normalize_images)

    # As we want to apply the masks, we iterate over the images and cut out the masked region of interest as bounding box
    if segmentation_path is not None:
        for i in range(len(images_1)):
            image1 = apply_mask(images_1[i], masks[i])
            images_1[i] = cv2.resize(image1, (224, 224), interpolation=cv2.INTER_LINEAR)
            image2 = apply_mask(images_2[i], masks[i])
            images_2[i] = cv2.resize(image2, (224, 224), interpolation=cv2.INTER_LINEAR)


    # Test if any of the images in images_1 is the same as images in images_2 based on hashed numpy arrays
    count = 0
    enumerator = len(images_1) if len(images_1) < 10 else 10
    for i in range(enumerator):
        for x2 in images_2:
            if str(images_1[i].tobytes()) == str(x2.tobytes()):
                count += 1
    if count > 1:
        print(
            f"Warning: Tested {enumerator} images and found {count} image pairs that are the same in both dataset. Is this expected? Revise the dataset.")


    num_batches = int(len(images_1) / 10) + 1 if len(images_1) / 10 > 1 else 1
    while num_batches > 1:
        if len(images_1) % num_batches == 0 and len(images_2) % num_batches == 0:
            break
        num_batches -= 1
    if num_batches <= 1:
        num_batches = 1
        print(
            f"Warning: batch size is 1, this might cause problems with memory when loading and comparing {len(images_1)} and {len(images_2)} images")

    print(f"Comparing {len(images_1)} vs {len(images_2)} images using num_batches={num_batches}.")

    img_array = np.array(images_1[0])
    min_val = img_array.min()
    max_val = img_array.max()
    print(np.array(images_1).shape)
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    # input1=tf.keras.applications.inception_v3.preprocess_input(images_1)
    # input1=tf.keras.applications.imagenet_utils.preprocess_input(images_1, mode="caffe")
    # input1=tf.keras.applications.resnet.preprocess_input(images_1)
    input1=images_1.astype('float32')/255
    # input2=tf.keras.applications.inception_v3.preprocess_input(images_2)
    # input2=tf.keras.applications.imagenet_utils.preprocess_input(images_2, mode="caffe")
    # input2=tf.keras.applications.resnet.preprocess_input(images_2)
    input2=images_2.astype('float32')/255
    img_array = np.array(input1[0])
    min_val = img_array.min()
    max_val = img_array.max()
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")

    with tf.device('/CPU:0'):
        fid = tfgan.eval.classifier_metrics.frechet_classifier_distance(
            input1,
            input2,
            # tf.keras.applications.inception_v3.preprocess_input(images_1),
            # tf.keras.applications.inception_v3.preprocess_input(images_2),
            get_classifier_fn(model_name),
            num_batches=num_batches)

    # Preprocess images and images to activations using CLF.
    #images_1 = _frechet_classifier_distance_helper(preprocess_input(images_1), get_classifier_fn(model_name), num_batches=num_batches)
    #images_2 = _frechet_classifier_distance_helper(preprocess_input(images_2), get_classifier_fn(model_name), num_batches=num_batches)
    #fid = tfgan.eval.classifier_metrics._frechet_classifier_distance_from_activations_helper(activations1=images_1, activations2=images_2, streaming=False)
    return fid

def apply_mask(image, mask):
    if len(mask) != 0 and len(mask[1]) != 0 and len(mask[0]) != 0:
        ymin, xmin, ymax, xmax = bounding_box(mask)[1]
        new_image = image[ymin:ymax, xmin:xmax]
        if np.count_nonzero(new_image) != 0: # skip this case as mask is empty
            image = new_image
    else:
        print(f"Mask is empty for this case.")
    return image

def bounding_box(mask):
    """Compute bounding boxes from masks.
    # https://discuss.pytorch.org/t/extracting-bounding-box-coordinates-from-mask/61179/4
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # Add 1 to the end coordinates to make them inclusive
            x2 += 1
            y2 += 1

        else:
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def _frechet_classifier_distance_helper(input_tensor1,
                                        classifier_fn,
                                        num_batches=1):
    """A helper function for evaluating the frechet classifier distance."""

    def compute_activations(elems):
        return tf.map_fn(
            fn=classifier_fn,
            elems=elems,
            parallel_iterations=1,
            back_prop=False,
            swap_memory=True,
            name='RunClassifier')
    return tf.concat(tf.unstack(compute_activations(tf.stack(tf.split(input_tensor1, num_or_size_splits=num_batches)))), 0)
    # return _frechet_classifier_distance_from_activations_helper(activations1, activations2, streaming=streaming)


def check_if_files_correspond(directory_1, directory_2, phase, secondphase, rename=True, enforce_strict_file_correspondence=True):

    if rename and ("_synthetic" in os.listdir(directory_1)[0] or "_synthetic" in os.listdir(directory_2)[0]):
        directories = [directory_1, directory_2]
        for directory in directories:
            for filename in os.listdir(directory):
                if '_synthetic' in filename:
                    filename1=filename.split('_synthetic')[0]
                    filename1=filename1+'.png'
                    my_dest = os.path.join(directory, filename1)
                else:
                    my_dest = os.path.join(directory, filename.replace("_synthetic", ""))
                my_source = os.path.join(directory, filename)
                os.rename(my_source, my_dest)

    file_names_1 = sorted(glob.glob(f'{directory_1}/*.png')) if ".png" in os.listdir(directory_1)[0] else sorted(glob.glob(f'{directory_1}/*.jpg'))
    file_names_2 = sorted(glob.glob(f'{directory_2}/*.png')) if ".png" in os.listdir(directory_2)[0] else sorted(glob.glob(f'{directory_2}/*.jpg'))
    file_names_without_path_1 = sorted(os.listdir(directory_1))
    file_names_without_path_2 = sorted(os.listdir(directory_2))
    print(f"Before enforcing correspondence: file_names_1[10] {file_names_1[10]}, file_names_2[10]: {file_names_2[10]}")

    print(f"file_names_without_path_2[100]: {file_names_without_path_2[80]}")
    print(f"file_names_without_path_2[80]: {file_names_without_path_2[80]}")

    if enforce_strict_file_correspondence:
        file_names_1_new = [file_name for file_name in file_names_1 if any(x in file_names_without_path_2 for x in get_file_transformations(os.path.basename(file_name), phase=phase, secondphase=secondphase))]
        file_names_2_new = [file_name for file_name in file_names_2 if any(x in file_names_without_path_1 for x in get_file_transformations(os.path.basename(file_name), phase=phase, secondphase=secondphase))]
        file_names_1 = file_names_1_new
        file_names_2 = file_names_2_new

    assert len(file_names_1) == len(file_names_2), f"Number of images in both datasets must be equal. {len(file_names_1)}!={len(file_names_2)}"
    assert len(file_names_1) != 0 or len(file_names_2) != 0, f"Number of file_names in a folder cannot be 0. Please revise. From {directory_1}: {len(file_names_1)}. From {directory_2}:{len(file_names_2)}"

    if not len(os.listdir(directory_1)) == len(os.listdir(directory_2)):
        print(f"Number of images in both datasets adjusted to {len(file_names_1)}. Initially number of images in {directory_1} and {directory_2} was not equal. {len(os.listdir(directory_1))}!={len(os.listdir(directory_2))}.")

    idx_for_checks = [0, 10, 30, int(len(file_names_1)/3), int(len(file_names_1)/2), len(file_names_1)-1]
    for idx in idx_for_checks:
        filename_1 = Path(os.fsdecode(file_names_1[idx])).name
        filename_2 = Path(os.fsdecode(file_names_2[idx])).name
        assert filename_1.replace("_mask", "_slice").replace("_synthetic", "").replace("_synthesized_image", "").replace(f"{phase}", f"{secondphase}").replace("jpg", "png") == filename_2.replace("_mask", "_slice").replace("_synthetic", "").replace("_synthesized_image", "").replace(f"{phase}", f"{secondphase}").replace("jpg", "png"), f"Files (at idx={idx}) do not correspond: {filename_1} and {filename_2}"

    return file_names_1, file_names_2

def get_file_transformations(file_name, phase, secondphase):
    transformed = [file_name, file_name.replace(f"_{phase}", f"_{secondphase}"), file_name.replace(f"_{secondphase}", f"_{phase}")]
    final_transformed = []
    for transformed_filename in transformed:
        final_transformed.extend([transformed_filename.replace("png", "jpg"), transformed_filename.replace("jpg", "png"), transformed_filename.replace("_synthetic.png", ".png"),   transformed_filename.replace("_slice", "_mask"), transformed_filename.replace("_mask", "_slice"), transformed_filename.replace(".png", "_synthetic.png")])
    return list(set(final_transformed))

if __name__ == "__main__":
    args = parse_args()
    print(f"args for FID computation: {args}")
    directory_1 = args.dataset_path_1
    directory_2 = args.dataset_path_2
    lower_bound = args.lower_bound
    normalize_images = args.normalize_images
    model_name = args.model
    is_split_per_patient = args.is_split_per_patient
    reverse_split_ds1 = args.reverse_split_ds1
    reverse_split_ds2 = args.reverse_split_ds2
    is_only_splitted_loaded = args.is_only_splitted_loaded
    limit = args.limit

    fid = calculate_fid(
        directory_1=directory_1,
        directory_2=directory_2,
        model_name=model_name,
        lower_bound=lower_bound,
        normalize_images=normalize_images,
        is_split_per_patient=is_split_per_patient,
        reverse_split_ds1=reverse_split_ds1,
        reverse_split_ds2=reverse_split_ds2,
        is_only_splitted_loaded=is_only_splitted_loaded,
        limit=limit,
        phase=args.phase,
        secondphase=args.secondphase,
        segmentation_path=args.segmentation_path,
    )

    if lower_bound:
        print("From {} samples (phase:{}, secondphase:{}), lower bound FID {}: {}".format(args.limit + 1, args.phase, args.secondphase, model_name, fid))
    else:
        print("From {} samples (phase:{}, secondphase:{}), FID {}: {}".format(args.limit + 1, args.phase, args.secondphase, model_name, fid))

    fid_results = [args.description, float(fid), model_name, args.limit + 1, f'normalised: {normalize_images}', directory_1,
                   directory_2, str(datetime.now())]

    # Open existing CSV file in append mode and add FID info
    with open('fid.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(fid_results)
        f_object.close()



