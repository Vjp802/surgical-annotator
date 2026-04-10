import numpy as np
from app.services.sam_service import SAMService
sam = SAMService()
sam.load_models()
image = np.zeros((100, 100, 3), dtype=np.uint8)
sam.segment_image(image, 50, 50)
print("Success!")
