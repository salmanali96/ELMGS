

import lzma
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import os
import time

def compress_point_cloud(input_file, output_file):
    # Read the input point cloud data
    with open(input_file, 'rb') as f_in:
        input_data = f_in.read()

    decompressing_time = []
    for i in range(0,3):
        start_time = time.time()
        lzma.decompress(input_data)
        end_time = time.time()
        decompress_time = end_time - start_time
        decompressing_time.append(decompress_time)

    avg_time = sum(decompressing_time)/3


    # Write the compressed data to the output file
    # with open(output_file, 'wb') as f_out:
    #     f_out.write(decompressed_data)

    print(f'Total time to decompress: {avg_time}')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="DeCompressing Quantized Ply ")
    parser.add_argument("--model_path", default='./output/with-eval/quantization-after-pruning/train/0.025', type=str)
    parser.add_argument("--iteration", default=71_000, type=int)
    args = get_combined_args(parser)
    print("DeCompressing.... " + args.model_path  + " at iteration: " + str(args.iteration))

    # Example usage:
    model_path = os.path.join(args.model_path, "point_cloud/iteration_{}".format(args.iteration), "point_cloud_quantized_decompressed.ply")
    output_path = os.path.join(args.model_path, "compress_ply/iteration_{}".format(args.iteration))
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path,  "point_cloud_quantized.xz")
    compress_point_cloud(output_path, model_path)