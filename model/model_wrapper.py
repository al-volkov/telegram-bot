import os
from io import BytesIO
import numpy as np
import torch
from PIL import Image

from model.msgnet import MSGNet


class ModelWrapper:
    def __init__(self):
        style_model = MSGNet()
        model_dict = torch.load(os.path.join("model", "final.model"))
        model_dict_clone = model_dict.copy()
        for key, value in model_dict_clone.items():
            if key.endswith(('running_mean', 'running_var')):
                del model_dict[key]
        style_model.load_state_dict(model_dict, False)
        self.model = style_model

    def process(self, style, content):
        original_width, original_height = content.size
        size2 = int(256.0 / content.size[0] * content.size[1])
        content = content.resize((256, size2), Image.LANCZOS)
        content = np.array(content).transpose((2, 0, 1))
        content = torch.from_numpy(content).float()
        content_image = content.unsqueeze(0)
        style = style.resize((512, 512), Image.LANCZOS)
        style = np.array(style).transpose((2, 0, 1))
        style = torch.from_numpy(style).float()
        style = style.unsqueeze(0)
        style = preprocess_batch(style)
        style_v = torch.Tensor(style)
        content_image = torch.Tensor(preprocess_batch(content_image))
        self.model.set_target(style_v)
        output = self.model(content_image)
        tensor = output.data[0]
        (b, g, r) = torch.chunk(tensor, 3)
        tensor = torch.cat((r, g, b))
        img = tensor.clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype('uint8')
        img = Image.fromarray(img)
        img = img.resize((original_width, original_height))
        bytes = BytesIO()
        bytes.name = 'result.jpeg'
        img.save(bytes, 'JPEG')
        bytes.seek(0)
        return bytes


def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch
