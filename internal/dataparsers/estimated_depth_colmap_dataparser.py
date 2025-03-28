import os
import json
import numpy as np
import torch
from dataclasses import dataclass
from .colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers import DataParserOutputs


@dataclass
class EstimatedDepthColmap(Colmap):
    depth_dir: str = "estimated_depths"

    depth_rescaling: bool = True

    depth_scale_name: str = "estimated_depth_scales"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.

    allow_depth_interpolation: bool = False

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "EstimatedDepthColmapDataParser":
        return EstimatedDepthColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class EstimatedDepthColmapDataParser(ColmapDataParser):
    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        if self.params.depth_rescaling is True:
            with open(os.path.join(self.path, self.params.depth_scale_name + ".json"), "r") as f:
                depth_scales = json.load(f)

            image_name_set = {image_name: True for image_name in dataparser_outputs.train_set.image_names + dataparser_outputs.val_set.image_names}
            depth_scale_list = []
            for image_name, image_depth_scale in depth_scales.items():
                if image_name not in image_name_set:
                    continue
                depth_scale_list.append(image_depth_scale["scale"])

            median_scale = np.median(np.asarray(depth_scale_list))

        loaded_depth_count = 0
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{image_name}.npy")
                if os.path.exists(depth_file_path) is False:
                    print("[WARNING] {} does not have a depth file".format(image_name))
                    continue

                depth_scale = {
                    "scale": 1.,
                    "offset": 0.,
                }
                if self.params.depth_rescaling is True:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        print("[WARNING {} does not have a depth scale]".format(image_name))
                        continue
                    if depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale:
                        print("[WARNING depth scale '{}' of '{}' out of bound '({}, {})']".format(
                            depth_scale["scale"],
                            image_name,
                            self.params.depth_scale_lower_bound * median_scale,
                            self.params.depth_scale_upper_bound * median_scale,
                        ))
                        continue

                image_set.extra_data[idx] = (depth_file_path, depth_scale, (image_set.cameras[idx].height.item(), image_set.cameras[idx].width.item()))
                loaded_depth_count += 1
            image_set.extra_data_processor = self.get_depth_loader(self.params.allow_depth_interpolation)

        assert loaded_depth_count > 0
        print("found {} depth maps".format(loaded_depth_count))

        return dataparser_outputs

    @staticmethod
    def get_depth_loader(allow_depth_interpolation: bool):
        def load_depth(depth_info):
            if depth_info is None:
                return None

            depth_file_path, depth_scale, image_shape = depth_info
            depth = np.load(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]
            depth = torch.tensor(depth, dtype=torch.float)
            depth = torch.clamp_min(depth, min=0.)

            if depth.shape != image_shape:
                assert allow_depth_interpolation, "the shape '{}' of depth map '{}' and '{}' of image not match, add the '--data.parser.allow_depth_interpolation=true' if you are sure this is expected".format(depth.shape, depth_file_path, image_shape)
                depth = torch.nn.functional.interpolate(
                    depth[None, None, ...],
                    image_shape,
                    mode="bilinear",
                    align_corners=True,
                )[0, 0]

            return depth

        return load_depth
