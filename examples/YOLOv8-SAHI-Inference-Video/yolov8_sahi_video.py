import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.ultralytics import download_yolo11n_model

from ultralytics.utils.plotting import Annotator, colors


class SAHIInference:
    def __init__(self):
        self.model = None

    def load_model(self, weights="yolo11n.pt"):
        # Check if the model file exists and load it
        download_yolo11n_model(weights)
        try:
            self.model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=weights, device="cpu"
            )
        except Exception as e:
            print("Error loading model:", e)
            raise

    


if __name__ == "__main__":
    inference = SAHIInference()
    inference.run()