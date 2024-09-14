import sys
import os
import time
from tqdm import tqdm
import torch
import yaml
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import onnxruntime
from deepsparse import compile_model
from ultralytics import YOLO
import logging
from statistics import mean, stdev
torch.set_float32_matmul_precision('high')
sys.path.append('./YOLOv8-test')
from nets import nn
from utils.util import non_max_suppression

global batch_size, class_no
batch_size = 1
class_no = 80

# Set up logging
logging.basicConfig(filename='yolo_comparison.log', level=logging.INFO,
                    format='- %(levelname)s - %(message)s')

def load_custom_model(weights_path, num_classes):
    model = nn.yolo_v8_n(num_classes).cpu()
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['model'].float().state_dict(), strict=False)
    model.eval()
    return model.fuse()

def preprocess_images(image_folder, input_size):
    resize_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    batches = []
    
    preprocess_times = []

    for i in range(0, len(image_files), batch_size):
        batch = []
        for j in range(i, min(i + batch_size, len(image_files))):
            start_time = time.time()
            image = Image.open(os.path.join(image_folder, image_files[j]))
            tensor = resize_transform(image)
            tensor = tensor / 255
            end_time = time.time()
            preprocess_times.append(end_time - start_time)
            
            batch.append(tensor)
        batches.append(torch.stack(batch))
    
    avg_preprocess_time = mean(preprocess_times)
    logging.info(f"Average Preprocessing Time per image ({input_size}x{input_size}): {avg_preprocess_time:.4f} seconds")
    print(f"Average Preprocessing Time per image ({input_size}x{input_size}): {avg_preprocess_time:.4f} seconds")
    return batches

def run_inference(model_func, batches, num_runs=5, num_warmup=5):
    # Warm-up runs
    for _ in range(num_warmup):
        model_func(batches[0])
    
    all_inference_times = []
    all_post_processing_times = []
    
    for run in range(num_runs):
        inference_times = []
        post_processing_times = []
        for batch in tqdm(batches, desc=f"Run {run+1}/{num_runs}"):
            start_time = time.time()
            pred = model_func(batch)
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            # Post-processing
            start_time = time.time()
            non_max_suppression(torch.tensor(pred[0]) if not isinstance(pred, torch.Tensor) else pred, classes=class_no)
            end_time = time.time()
            post_processing_times.append(end_time - start_time)
        
        all_inference_times.extend(inference_times)
        all_post_processing_times.extend(post_processing_times)
    
    avg_inference_time = mean(all_inference_times)
    avg_post_processing_time = mean(all_post_processing_times)
    throughput = 1 / avg_inference_time
    
    return avg_inference_time, avg_post_processing_time, throughput, stdev(all_inference_times)

def inference_pytorch_cpu(model, batches, num_runs=5, num_warmup=5):
    def model_func(batch):
        with torch.no_grad():
            return model(batch)
    
    return run_inference(model_func, batches, num_runs, num_warmup)

def inference_onnx(onnx_path, batches, num_runs=5, num_warmup=5):
    session = onnxruntime.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def model_func(batch):
        return session.run(None, {input_name: batch.numpy()})
    
    return run_inference(model_func, batches, num_runs, num_warmup)

def inference_deepsparse(onnx_path, batches, num_runs=5, num_warmup=5):
    pipe = compile_model(onnx_path, batch_size=batch_size)

    def model_func(batch):
        return pipe([batch.numpy()])
    
    return run_inference(model_func, batches, num_runs, num_warmup)

def inference_ultralytics(model, batches, num_runs=5, num_warmup=5):
    def model_func(batch):
        return model.model(batch)
    
    return run_inference(model_func, batches, num_runs, num_warmup)

def run_comparisons(input_size, num_runs=3):
    logging.info(f"\nRunning comparisons for {input_size}x{input_size} images:")
    print(f"\nRunning comparisons for {input_size}x{input_size} images:")
    # Prepare data
    batches = preprocess_images('./datasets/coco128/images/train2017', input_size)

    # Your custom YOLOv8 implementations
    dummy_input = torch.randn(1, 3, input_size, input_size)  
    custom_model = load_custom_model('./YOLOv8-test/weights/v8_n(1).pt', class_no)
    
    compiled_model = torch.compile(custom_model)
    
    """
    onnx_path_custom = f'./YOLOv8-test/weights/yolov8_custom_{input_size}.onnx'
    torch.onnx.export(custom_model, 
                  dummy_input, 
                  onnx_path_custom,
                  opset_version=13,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    os.system(f'onnxslim {onnx_path_custom} {onnx_path_custom}')
    """

    ultralytics_model = YOLO('yolov8n.pt')
    ultralytics_model.fuse()
    ultralytics_model.export(format="onnx", batch=batch_size,
                             imgsz=input_size,simplify=True, opset=13)
    prunned_path = "./model.onnx"
    # Run inferences
    #os.system(f'onnxslim {prunned_path} {prunned_path}')
    deepsparse_pruned = inference_deepsparse(prunned_path, batches, num_runs)

    pytorch_cpu_results = inference_pytorch_cpu(compiled_model, batches, num_runs)
    onnx_results = inference_onnx(onnx_path_custom, batches, num_runs)
    deepsparse_results = inference_deepsparse(onnx_path_custom, batches, num_runs)
    ultralytics_results = inference_ultralytics(ultralytics_model, batches, num_runs)
    ultralytics_onnx_results = inference_onnx(f'./yolov8n.onnx', batches, num_runs)
    ultralytics_deepsparse_results = inference_deepsparse(f'./yolov8n.onnx', batches, num_runs)

    return {
        'Pruned_deepsparse': deepsparse_pruned,
        'pytorch_cpu': pytorch_cpu_results,
        'onnx': onnx_results,
        'deepsparse': deepsparse_results,
        'ultralytics': ultralytics_results,
        'ultralytics_onnx': ultralytics_onnx_results,
        'ultralytics_deepsparse': ultralytics_deepsparse_results
    }

def main():
    with open('./YOLOv8-test/utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    num_runs = 5  # Number of full runs for each model and input size

    results_640 = run_comparisons(640, num_runs)
    results_256 = run_comparisons(256, num_runs)
    logging.info("---Compiled---")
    logging.info("\nComparison Results:")
    logging.info("640x640 Images:")
    print("\nComparison Results:")
    print("640x640 Images:")
    for model, results in results_640.items():
        log_message = (f"\n{model}: Inference Time = {results[0]:.4f} s ± {results[3]:.4f} s,\n"
                       f"Post-processing Time = {results[1]:.4f} s, Throughput = {results[2]:.2f} images/s\n")
        logging.info(log_message)
        print(log_message)
    
    logging.info("\n256x256 Images:")
    print("\n256x256 Images:")
    for model, results in results_256.items():
        log_message = (f"\n{model}: Inference Time = {results[0]:.4f} s ± {results[3]:.4f} s,\n"
                       f"Post-processing Time = {results[1]:.4f} s, Throughput = {results[2]:.2f} images/s\n")
        logging.info(log_message)
        print(log_message)

if __name__ == "__main__":
    main()