import torch, numpy as np
from PIL import Image
from app.services.sam_service import SAMService

sam = SAMService()
sam.load_models()
image = np.zeros((100, 100, 3), dtype=np.uint8)
pil = Image.fromarray(image)

try:
    state = sam.image_processor.set_image(pil)
    sam.image_processor.reset_all_prompts(state)
    state = sam.image_processor.add_geometric_prompt(box=[0.5,0.5,0.02,0.02], label=True, state=state)
    print('Success!')
except Exception as e:
    import traceback
    traceback.print_exc()
