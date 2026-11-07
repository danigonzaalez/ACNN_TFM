import argparse
import json
import contextlib
import os
from datetime import datetime
import numpy as np

from prepared_models.mobilenetv4 import main as mobilenetv4
from utils.torch_utils import accuracy_mean_deviation
from utils.general_utils import calculate_seg_mean_dev, NumpyEncoder
# import plot_results as plr   # utilities for plotting AD graphics using results file

model_functions = {
    "mobilenetv4": mobilenetv4,
}


def main(config):
    """Program for testing models over a complete dataset directory
    args:       config: dictionary containing directories and parameters
    return:     None
    """

    # Save the results in a dictionary
    results = {}

    EXP_NUMBER = config.get("EXP_NUMBER", 1)
    models = config.get("models", ["mobilenetv4"])

    print(f"EXP_NUMBER: {EXP_NUMBER}")
    print(f"Train Directory: {config.get('train_dir')}")
    print(f"Train GT Directory: {config.get('train_gt_dir')}")
    print(f"Test Directory: {config.get('test_dir')}")
    print(f"Test GT Directory: {config.get('test_gt_dir')}")

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
                        result = model_function(EXP_NUMBER, config)

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
            # Calculate the mean and deviation of the Segmentation results
            mean_seg_stats = calculate_seg_mean_dev(seg_stats)
            # Add the Segmentation results to the global results
            results[model]["Seg_results"] = mean_seg_stats

    # Save the results in a file
    # Create results folder if it does not exist
    results_folder = os.path.join("results")
    os.makedirs(results_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_folder, f"results_{timestamp}.json")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    # Print clean summary of the Segmentation results (this could go on a function in plot_results.py)
    print("\nSummary of the Segmentation results:")
    print(f"{'Model':<20} {'OA (%)':<15} {'AA (%)':<15} {'mIoU (%)':<15} {'fwIoU (%)':<15}")
    for model_name in models:
        if model_name in results:
            m_OA = np.mean(results[model_name]["OA"])
            m_AA = np.mean(results[model_name]["AA"])
            
            seg_res = results[model_name].get("Seg_results", {})
            
            # general_utils.calculate_seg_mean_dev creates {"mean": X, "std_dev": Y}
            m_mIoU = seg_res.get("mIoU", {}).get("mean", 0)
            m_fwIoU = seg_res.get("fwIoU", {}).get("mean", 0)
            
            print(f"{model_name:<20} {m_OA:<15.2f} {m_AA:<15.2f} {m_mIoU:<15.2f} {m_fwIoU:<15.2f}")

    # Plot the results (COMENTADO: eran gráficos heredados de Anomaly Detection)
    # plr.plot_results(results_file, config=config)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="List of models to test")
    parser.add_argument("--config", type=str, help="Path to the config file")

    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        config = {}
        
    # Override models in config if passed by command line
    if args.models:
        config["models"] = args.models

    # Launch main with the config
    main(config)