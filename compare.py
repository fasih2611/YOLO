import gc
import sys
import os
import time
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import onnxruntime
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import logging
from statistics import mean, stdev
torch.set_float32_matmul_precision('high')
sys.path.append('./YOLOv8-test')
from nets import nn
from utils.util import non_max_suppression
from torch.nn.utils import prune
import copy
import csv
import intel_extension_for_pytorch as ipex

global batch_size, class_no, preprocess_time
batch_size = 1
class_no = 80
preprocess_time = 0

class YOLODataset(Dataset):
    def __init__(self, image_folder, input_size):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])
        self.input_size = input_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        start_time = time.perf_counter()
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(image_path)
        tensor = self.transform(image)
        tensor = tensor / 255
        end_time = time.perf_counter()
        return tensor, end_time - start_time

def write_results_to_csv(results, filename='yolo_comparison_results.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Image Size', 'Preprocess Time', 'Inference Time', 'Post-processing Time', 'Throughput', 'Inference Time Std Dev']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for image_size, size_results in results.items():
            for model, model_results in size_results.items():
                writer.writerow({
                    'Model': model,
                    'Image Size': image_size,
                    'Preprocess Time': f'{model_results[0]:.4f}',
                    'Inference Time': f'{model_results[1]:.4f}',
                    'Post-processing Time': f'{model_results[2]:.4f}',
                    'Throughput': f'{model_results[3]:.4f}',
                    'Inference Time Std Dev': f'{model_results[4]:.4f}'
                })

logging.basicConfig(filename='yolo_comparison.log', level=logging.INFO,
                    format='- %(levelname)s - %(message)s')

def load_custom_model(weights_path, num_classes):
    model = nn.yolo_v8_n(num_classes).cpu()
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    model.eval()
    return model.fuse()

def sparsity(model):
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a

def safe_prune(module, name, amount):
    try:
        if isinstance(getattr(module, name), torch.nn.Parameter):
            prune.l1_unstructured(module, name=name, amount=amount)
            prune.remove(module, name)
            return True
    except Exception as e:
        print(f"Skipping pruning for {name} in {module.__class__.__name__}: {str(e)}")
    return False

def prune_layer(layer, amount=0.2):
    pruned = False
    if isinstance(layer, torch.nn.Conv2d):
        new_layer = copy.deepcopy(layer)        
        if safe_prune(new_layer, 'weight', amount):
            pruned = True
        if new_layer.bias is not None and safe_prune(new_layer, 'bias', amount):
            pruned = True
        if pruned:
            layer.weight.data = new_layer.weight.data
            if layer.bias is not None:
                layer.bias.data = new_layer.bias.data
    return pruned

def prune_model(model, amount=0.2):
    pruned_layers = 0
    total_layers = 0

    def recursive_prune(module):
        nonlocal pruned_layers, total_layers
        for name, child in module.named_children():
            if list(child.children()):
                recursive_prune(child)
            else:
                total_layers += 1
                if prune_layer(child, amount):
                    pruned_layers += 1
                    print(f"Pruned layer: {name} ({child.__class__.__name__})")
    
    recursive_prune(model)
    print(f"Pruned {pruned_layers} out of {total_layers} layers")
    return model

def create_dataloader(image_folder, input_size):
    dataset = YOLODataset(image_folder, input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Calculate average preprocessing time
    preprocess_times = []
    for _, batch_times in dataloader:
        preprocess_times.extend(batch_times.tolist())
    
    global preprocess_time
    preprocess_time = mean(preprocess_times)
    print(f"Average Preprocessing Time per image ({input_size}x{input_size}): {preprocess_time:.4f} seconds")
    
    return dataloader

def run_inference(model_func, dataloader, num_runs=5, num_warmup=5):
    # Get first batch for warmup
    first_batch = next(iter(dataloader))[0]
    
    # Warm-up runs
    for _ in range(num_warmup):
        model_func(first_batch)
    
    all_inference_times = []
    all_post_processing_times = []
    
    for run in range(num_runs):
        inference_times = []
        post_processing_times = []
        for batch, _ in tqdm(dataloader, desc=f"Run {run+1}/{num_runs}"):
            start_time = time.perf_counter()
            pred = model_func(batch)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            
            # Post-processing
            start_time = time.perf_counter()
            non_max_suppression(torch.tensor(pred[0]) if not isinstance(pred, torch.Tensor) else pred)
            end_time = time.perf_counter()
            post_processing_times.append(end_time - start_time)
        
        all_inference_times.extend(inference_times)
        all_post_processing_times.extend(post_processing_times)
    
    global preprocess_time
    avg_inference_time = mean(all_inference_times)
    avg_post_processing_time = mean(all_post_processing_times)
    throughput = 1 / (avg_inference_time + avg_post_processing_time + preprocess_time) 
    
    return preprocess_time, avg_inference_time, avg_post_processing_time, throughput, stdev(all_inference_times)

def inference_pytorch_cpu(model, dataloader, num_runs=5, num_warmup=5):
    def model_func(batch):
        with torch.no_grad():
            return model(batch)
    
    return run_inference(model_func, dataloader, num_runs, num_warmup)

def inference_onnx(onnx_path, dataloader, num_runs=5, num_warmup=5):
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def model_func(batch):
        return session.run(None, {input_name: batch.numpy()})
    
    return run_inference(model_func, dataloader, num_runs, num_warmup)

def inference_ultralytics(model, dataloader, num_runs=5, num_warmup=5):
    def model_func(batch):
        return model.model(batch)
    
    return run_inference(model_func, dataloader, num_runs, num_warmup)

def run_comparisons(input_size, model_type, num_runs=3):
    print(f"\nRunning comparisons for {input_size}x{input_size} images:")
    # Prepare data
    dataloader = create_dataloader('./datasets/coco128/images/train2017', input_size)

    ultralytics_model = YOLO(f'{model_type}.pt')
    ultralytics_model.fuse()

    ipex_model = YOLO(f'{model_type}.pt')
    ipex_model.fuse()
    ipex_model = ipex_model.model
    ipex_model.compile(backend='ipex')
    
    ultralytics_model.export(format="onnx", batch=batch_size,
                             imgsz=input_size, simplify=True, opset=13)
    
    # Run inferences
    ultralytics_ipex_results = inference_pytorch_cpu(ipex_model, dataloader, num_runs)
    gc.collect()
    ultralytics_results = inference_ultralytics(ultralytics_model, dataloader, num_runs)
    gc.collect()
    ultralytics_onnx_results = inference_onnx(f'./{model_type}.onnx', dataloader, num_runs)
    gc.collect()
    
    del ultralytics_model

    return {
        f'{model_type} ultralytics': ultralytics_results,
        f'ultralytics_ipex': ultralytics_ipex_results,
        f'{model_type} ultralytics_onnx': ultralytics_onnx_results,
    }

def main():
    num_runs = 5
    sizes = [1024, 768, 640, 320, 128]
    models = ['yolov5n6u', 'yolov8n', 'yolov9t', 'yolov10n', 'yolo11n']
    results = {}
    for model in models:
        for size in sizes:
            results[size] = run_comparisons(size, model, num_runs)
        write_results_to_csv(results, filename=f'{model} - speed comparison')

if __name__ == "__main__":
    main()