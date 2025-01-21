
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os
import pickle
import json
import lzma
from plyfile import PlyData, PlyElement
import numpy as np
def load_ply(path, scene_path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    save_dict = dict()
    save_dict["xyz"] = xyz
    save_dict["opacities"] = opacities
    save_dict["features_dc"] = features_dc
    save_dict["features_extra"] = features_extra
    save_dict["scales"] = scales
    save_dict["rots"] = rots
    save_dict["path"] = path

    size = {}

    size["compress_without_morton"] = compress_without_morton(xyz, opacities, features_dc, features_extra, scales, rots)

    xyz, opacities, features_dc, features_extra, scales, rots = _sort_morton(xyz, opacities, features_dc, features_extra, scales, rots, path)

    size["compress_with_morton"] = compress_without_morton(xyz, opacities, features_dc, features_extra, scales, rots)



    # Open the file in write mode and use json.dump() to write the data to the file
    with open(scene_path + "/compression_result.json", 'w') as fp:
        json.dump(size, fp, indent=True)

    return xyz, opacities, features_dc, features_extra, scales, rots, path



def compress_without_morton(xyz, opacities, features_dc, features_extra, scales, rots):

    combined_data = {'xyz': xyz,
                     'opacities': opacities,
                     'features_dc': features_dc,
                     'features_extra': features_extra,
                     'scales': scales,
                     'rots': rots
    }

    serialized_data = pickle.dumps(combined_data)

    compressed_data = lzma.compress(serialized_data)

    compressed_size_bytes = len(compressed_data)

    compressed_size_mb = compressed_size_bytes / (1024 * 1024)  # Convert bytes to MB

    print(f"Size of the compressed data: {compressed_size_mb:.2f} MB")

    return compressed_size_mb

def _sort_morton(xyz, opacities, features_dc, features_extra, scales, rots, path):


            min_values = xyz.min(axis=0)
            max_values = xyz.max(axis=0)

            xyz_q = (
                    (2 ** 21 - 1)
                    * (xyz - min_values)
                    / (max_values - min_values)
            ).astype(np.int64)

            morton_codes = mortonEncode(xyz_q)
            order = np.argsort(morton_codes)

            xyz = xyz[order]
            opacities = opacities[order]
            scales = scales[order]
            features_extra = features_extra[order]
            features_dc = features_dc[order]
            rots = rots[order]

            return xyz, opacities, features_dc, features_extra, scales, rots


def splitBy3(a):
    x = a & 0xFFFFFFFF  # we only look at the first 32 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x



def mortonEncode(pos: np.ndarray) -> np.ndarray:
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    answer = np.zeros(len(pos), dtype=np.int64)
    answer |= splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2)
    return answer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Restoring Quantized Ply ")
    parser.add_argument("--model_path", default='./output/gradient-aware/QAT/db/playroom/0.5', type=str)
    parser.add_argument("--iteration", default=90_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Compressing: " + args.model_path)

    model_path_ply = os.path.join(args.model_path, "point_cloud/iteration_{}".format(args.iteration),
                              "point_cloud_quantized.ply")
    load_ply(model_path_ply, args.model_path)









