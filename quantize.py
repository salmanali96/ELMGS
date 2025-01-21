import numpy as np
from plyfile import PlyData, PlyElement
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os

def load_ply(path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, opacities, features_dc, features_extra, scales,rots, path
        #save_ply(xyz, opacities, features_dc, features_extra, scales,rots, path = 'output.ply')
        #save_ply(np.round(xyz).astype(np.int8), np.round(opacities).astype(np.int8), np.round(features_dc).astype(np.int8), np.round(features_extra).astype(np.int8), np.round(scales).astype(np.int8),np.round(rots).astype(np.int8), path = 'quantize_output.ply')
        #save_ply(xyz, opacities, np.round(features_dc).astype(np.int32), np.round(features_extra).astype(np.int32), scales, rots, path = 'quantize_output_ag.ply')
        save_ply(np.round(xyz).astype(np.int16), opacities, np.round(features_dc).astype(np.int16), np.round(features_extra).astype(np.int16), np.round(scales).astype(np.int16),np.round(rots).astype(np.int16), path = 'quantize_output_16bit.ply')


def save_ply(xyz, opacities, features_dc, features_extra, scale,rots, path):
    
        normals = np.zeros_like(xyz)

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(xyz, opacities, features_dc, features_extra, scale,rots)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        features_dc = np.reshape(features_dc,(xyz.shape))
        features_extra = np.reshape(features_extra,((xyz.shape[0],-1)))

        attributes = np.concatenate((xyz, normals, features_dc, features_extra, opacities, scale, rots), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

def construct_list_of_attributes(xyz, opacities, features_dc, features_extra, scale,rots):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_extra.shape[1]*features_extra.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scale.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rots.shape[1]):
            l.append('rot_{}'.format(i))
        return l


def restore_ply(path, scales_path, output_path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        f_dc_scale = load_scale(scales_path, '_f_dc.pt')
        f_dc_scale = f_dc_scale.detach().cpu().numpy()

        f_rest_scale= load_scale(scales_path, '_f_rest.pt')
        f_rest_scale = f_rest_scale.detach().cpu().numpy()

        rotation_scale = load_scale(scales_path, '_rotation.pt')
        rotation_scale = rotation_scale.detach().cpu().numpy()

        scaling_scale = load_scale(scales_path, '_scale.pt')
        scaling_scale = scaling_scale.detach().cpu().numpy()

        xyz_scale = load_scale(scales_path, '_xyz.pt')
        xyz_scale = xyz_scale.detach().cpu().numpy()

        opacities_scale = load_scale(scales_path, '_opacities.pt')
        opacities_scale = opacities_scale.detach().cpu().numpy()

        xyz = xyz * xyz_scale
        features_dc = features_dc * f_dc_scale
        features_extra = features_extra * f_rest_scale
        scales = scales * scaling_scale
        rots = rots * rotation_scale
        opacities = opacities * opacities_scale

        #return xyz, opacities, features_dc, features_extra, scales,rots, path

        save_ply(xyz, opacities, features_dc, features_extra, scales,rots, path = output_path)

def load_scale(scale_path, name):
    scales_path = os.path.join(scale_path, name)
    return torch.load(scales_path)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Restoring Quantized Ply ")
    parser.add_argument("--model_path", default='./output/with-eval/quantization-after-pruning/train/0.025', type=str)
    parser.add_argument("--iteration", default=71_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    model_path = os.path.join(args.model_path, "point_cloud/iteration_{}".format(args.iteration), "point_cloud_quantized.ply")
    scales_path = os.path.join(args.model_path, "scales/iteration_{}".format(args.iteration))
    output_directory = os.path.join(args.model_path, "point_cloud/iteration_100000")
    output_path = os.path.join(args.model_path, "point_cloud/iteration_100000", "point_cloud.ply")
    os.makedirs(output_directory, exist_ok=True)
    restore_ply(model_path, scales_path, output_path)




