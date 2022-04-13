from sahi.model import MmdetDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict

model_type = "mmdet"
model_path = "/massStorage/disc1/yf/mmlab/work_dirs/swint_3x_800-1400_anchor_bs1x2/epoch_15.pth"
model_config_path = "/massStorage/disc1/yf/mmlab/work_dirs/swint_3x_800-1400_anchor_bs1x2/swint_test.py"
model_device = "cuda:2" # or 'cuda:0'
model_confidence_threshold = 0.4

slice_height = 256
slice_width = 256
overlap_height_ratio = 0.2
overlap_width_ratio = 0.2

source_image_dir = "/massStorage/disc1/yf/mmlab/data/logdet/val/images/"
dataset_json_path = "/massStorage/disc1/yf/mmlab/data/logdet/val/annotations/instances_val2017.json"

predict(
    model_type=model_type,
    model_path=model_path,
    model_config_path=model_config_path,
    model_device=model_device,
    model_confidence_threshold=model_confidence_threshold,
    source=source_image_dir,
    dataset_json_path=dataset_json_path,
    slice_height=slice_height,
    slice_width=slice_width,
    overlap_height_ratio=overlap_height_ratio,
    overlap_width_ratio=overlap_width_ratio,
)


