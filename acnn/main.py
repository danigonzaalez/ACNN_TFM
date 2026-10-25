import argparse
import json

from prepared_models.cnn21_seg import main as cnn21_seg
from prepared_models.acnn import main as acnn
from prepared_models.mobilenetv4 import main as mobilenetv4


from utils.torch_utils import accuracy_mean_deviation
from utils.general_utils import calculate_ad_mean_dev, NumpyEncoder

import contextlib
import os
from datetime import datetime

import plot_results as plr

model_functions = {
    "cnn21_seg": cnn21_seg,
    "acnn": acnn,
    "mobilenetv4": mobilenetv4,
}


def main(image_path, gt_path, models, config):
    """Program for testing multiple models in the same image
    args:   image_path: path to the image
            gt_path: path to the ground truth
            models: list of models to test
    return: None

    """

    # Save the results in a dictionary
    results = {}

    EXP_NUMBER = config.get("EXP_NUMBER", 1)
    image_path = config.get("image_path", image_path)
    gt_path = config.get("gt_path", gt_path)
    seg_path = config.get("seg_path", None)
    centers_path = config.get("centers_path", None)
    models = config.get("models", models)

    print(f"EXP_NUMBER: {EXP_NUMBER}")
    print(f"image_path: {image_path}")
    print(f"gt_path: {gt_path}")
    print(f"seg_path: {seg_path}")
    print(f"centers_path: {centers_path}")

    # for model in tqdm(models, desc="Processing models"):
    # Initialize metrics for averaging

    for model in models:
        print(f"[{models.index(model) + 1}/{len(models)}] Testing model {model}")
        OA = [0] * EXP_NUMBER
        AA = [0] * EXP_NUMBER
        aa = [0] * EXP_NUMBER
        seg_stats = [{}] * EXP_NUMBER
        for exp in range(EXP_NUMBER):
            print(f"Experiment {exp + 1}/{EXP_NUMBER} for model {model}")
            try:
                model_function = model_functions[model]
                # Create log folder if it does not exist
                os.makedirs("log", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"log/results_{model}_{timestamp}.txt", "w") as f:
                    with contextlib.redirect_stdout(f):
                        result = model_function(
                            EXP_NUMBER,
                            image_path,
                            gt_path,
                            seg_path,
                            centers_path,
                            config,
                        )
                        # Unpack based on the length of the result
                        if len(result) == 4:
                            OA[exp], AA[exp], aa[exp], seg_stats[exp] = result
                        else:
                            OA[exp], AA[exp], aa[exp] = result
                            seg_stats[exp] = None  # Set to None if not present
            except KeyError:
                print(f"Model {model} not found. Skipping...")
                # Skip to the next model
                continue
            except Exception as e:
                print(f"An error occurred while testing model {model}: {e}")
                # print traceback
                import traceback

                traceback.print_exc()
                # Skip to the next model
                continue

        # Add the evaluation metrics to the global results
        OAm, AAm, OAd, AAd, aam, aad = accuracy_mean_deviation(OA, AA, aa)
        results[model] = {
            "OA": OAm,
            "AA": AAm,
            "class_aa": aam,
            "OA_dev": OAd,
            "AA_dev": AAd,
            "class_aa_dev": aad,
        }

        if seg_stats[0] is not None:
            # Calculate the mean and deviation of the AD results
            mean_ad_results = calculate_ad_mean_dev(seg_stats)

            # Add the AD results to the global results
            results[model]["Seg_results"] = mean_ad_results

    # Print the results
    # print("\nResults:")
    # print(json.dumps(results, indent=4, cls=NumpyEncoder))
    # print("\n")

    # Save the results in a file
    # Create results folder if it does not exist
    results_folder = os.path.join("results")
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_folder, f"results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    # Plot the results
    plr.plot_results(results_file, config=config)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to the image")
    parser.add_argument("--gt", type=str, help="Path to the ground truth")
    parser.add_argument("--models", nargs="+", help="List of models to test")
    parser.add_argument("--config", type=str, help="Path to the config file")

    if parser.parse_args().config:
        config = json.load(open(parser.parse_args().config))
    else:
        config = {}

    main(
        parser.parse_args().image,
        parser.parse_args().gt,
        parser.parse_args().models,
        config,
    )
