import torch
import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score

def save_prediction(result, save_dir, mode="predict"):
    if mode == "predict":
        flat_outputs_default = defaultdict(list)
        for output in result:
            for key in output.keys():
                flat_outputs_default[key] += output[key]

        save_file = Path(save_dir) / "prediction_output.pt"
        torch.save(dict(flat_outputs_default), save_file)
        print(f"Predition result saved in {save_file}")

    elif mode == "patch":
        flat_outputs_default = defaultdict(list)
        for output in result:
            for key in output.keys():
                flat_outputs_default[key] += output[key]

        predictions = []
        targets = []
        file_patches = defaultdict(list)
        for output in result:
            predictions += output["prediction"]
            targets += output["target"]
            for key in ["filename", "top_patches"]:
                file_patches[key] += output[key]

        predictions = torch.stack(predictions).detach().cpu().numpy()
        targets = torch.stack(targets).detach().cpu().numpy()
        mAP = average_precision_score(targets, predictions, average="macro")
        print(f"mAP:   {mAP}")

        save_file = "/mnt/lwy/amu/tasks/audioset/top_patches/t.pt"
        torch.save(dict(file_patches), save_file)
        print(f"Top patches saved in {save_file}")
    
    else:
        with open(Path(save_dir) / "test_output.json", "w") as f:
            json.dump(result[0], f)
        