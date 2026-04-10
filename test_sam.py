import torch
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

try:
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device="mps")
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[100, 100]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    print("Scores:", scores)
except Exception as e:
    import traceback
    traceback.print_exc()
