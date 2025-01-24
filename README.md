
# [WACV 2025] ELMGS: Enhancing memory and computation scaLability through coMpression for 3D Gaussian Splatting



This code is build upon the official implementation of the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering". To set up the code, please refer to the original repository





For training the baseline model, please use the following command
```shell
python train_baseline.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path for the model to be saved>
```

For training the model with our ELMGS

Gradient and Opacity Aware Pruning
```shell
python prunning_play.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path for the model to be saved> --pruning_level <gamma_iter value> --start_checkpoint <path to pre-trained model> --iteration <number of iteration to train>
```
LSQ based Quantization Aware Training (QAT)
```shell
python QAT.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path for the model to be saved> --pruning_status no --quantization_status yes --start_checkpoint <path to pre-trained model of GAP>   --iteration <number of iteration to train> --checkpoint_quantization_iteration <should be same as last point cloud save iteration>
```

```shell
python compress.py --model_path <path to quantized trained model> --iteration <iteration of quantized point cloud>
```


For evaluating the model 
```shell
python render.py -m <path to trained model> --iteration <the iteration number at which the model is to be loaded> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

For evaluating FPS
```shell
python fps.py -m <path to trained model> --iteration <the iteration number at which the model is to be loaded> # Calculate FPS
```
