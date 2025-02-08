import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


def generate_random_image(image_size: int) -> np.ndarray:
    """
    Generate a random grayscale test image of specified size.

    Args:
        image_size: Width and height of the square image to generate

    Returns:
        np.ndarray: Random grayscale image of shape (image_size, image_size)
    """
    return np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)


def compute_hog_features_cpu(
        image: np.ndarray,
        window_size: Tuple[int, int],
        block_size: Tuple[int, int],
        block_stride: Tuple[int, int],
        cell_size: Tuple[int, int],
        histogram_bins: int
) -> Tuple[np.ndarray, float]:
    """
    Compute HOG features using CPU implementation.

    Args:
        image: Input grayscale image
        window_size: Detection window size (width, height)
        block_size: Block size for normalization (width, height)
        block_stride: Stride between blocks (width, height)
        cell_size: Cell size for histogram computation (width, height)
        histogram_bins: Number of histogram bins

    Returns:
        Tuple containing:
        - HOG descriptor array
        - Computation time in seconds
    """
    hog_descriptor = cv2.HOGDescriptor(
        window_size,
        block_size,
        block_stride,
        cell_size,
        histogram_bins
    )

    start_time = time.time()
    features = hog_descriptor.compute(image)
    computation_time = time.time() - start_time

    return features, computation_time


def compute_hog_features_gpu(
        image: np.ndarray,
        window_size: Tuple[int, int],
        block_size: Tuple[int, int],
        block_stride: Tuple[int, int],
        cell_size: Tuple[int, int],
        histogram_bins: int
) -> Tuple[np.ndarray, float]:
    """
    Compute HOG features using GPU implementation.

    Args:
        image: Input grayscale image
        window_size: Detection window size (width, height)
        block_size: Block size for normalization (width, height)
        block_stride: Stride between blocks (width, height)
        cell_size: Cell size for histogram computation (width, height)
        histogram_bins: Number of histogram bins

    Returns:
        Tuple containing:
        - HOG descriptor array
        - Computation time in seconds (including memory transfers)
    """
    hog_descriptor = cv2.cuda.HOG_create(
        win_size=window_size,
        block_size=block_size,
        block_stride=block_stride,
        cell_size=cell_size,
        nbins=histogram_bins
    )

    compute_time_only = True

    if compute_time_only:
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # Start timing (includes memory transfers)
        start_time = time.time()

        gpu_features = hog_descriptor.compute(gpu_image)

        computation_time = time.time() - start_time

        # Transfer results back to CPU
        cpu_features = gpu_features.download()

    else:
        # Start timing (includes memory transfers)
        start_time = time.time()

        # Transfer image to GPU
        gpu_image = cv2.cuda_GpuMat()
        gpu_image.upload(image)

        # Compute HOG features
        # This cause an error : AttributeError: 'cv2.cuda.HOG' object has no attribute 'computeBlockHistograms'
        # gpu_features = hog_descriptor.computeBlockHistograms(gpu_image)

        gpu_features = hog_descriptor.compute(gpu_image)

        # Transfer results back to CPU
        cpu_features = gpu_features.download()
        computation_time = time.time() - start_time

    return cpu_features, computation_time


def visualize_performance_results(
        image_sizes: List[int],
        cpu_times: List[float],
        gpu_times: List[float],
        speedups: List[float],
        save_path: str = None
) -> None:
    """
    Create visualization plots for HOG performance comparison.

    Args:
        image_sizes: List of image sizes tested
        cpu_times: List of average CPU computation times
        gpu_times: List of average GPU computation times
        speedups: List of CPU/GPU speedup ratios
        save_path: Optional path to save the plot as image file
    """
    plt.figure(figsize=(15, 5))

    # Execution time comparison plot
    plt.subplot(1, 2, 1)
    plt.plot(image_sizes, cpu_times, 'b-', marker='o', label='CPU')
    plt.plot(image_sizes, gpu_times, 'r-', marker='o', label='GPU')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('HOG Computation Time vs Image Size')
    plt.legend()
    plt.grid(True)

    # Speedup plot
    plt.subplot(1, 2, 2)
    plt.plot(image_sizes, speedups, 'g-', marker='o')
    plt.xlabel('Image Size (pixels)')
    plt.ylabel('Speedup Factor (CPU time / GPU time)')
    plt.title('GPU Speedup vs Image Size')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def benchmark_hog_performance(
        min_size: int = 128,
        max_size: int = 1024,
        size_step: int = 128,
        num_iterations: int = 5
) -> Dict[str, List[Any]]:
    """
    Benchmark HOG feature computation performance for CPU and GPU implementations.

    Args:
        min_size: Minimum image size to test
        max_size: Maximum image size to test
        size_step: Step size between test image sizes
        num_iterations: Number of iterations for each size for reliable timing

    Returns:
        Dictionary containing test results:
        - 'sizes': List of image sizes tested
        - 'cpu_times': List of average CPU computation times
        - 'gpu_times': List of average GPU computation times
        - 'speedups': List of CPU/GPU speedup ratios
    """
    # HOG parameters
    hog_params = {
        'window_size': (64, 128),
        'block_size': (16, 16),
        'block_stride': (8, 8),
        'cell_size': (8, 8),
        'histogram_bins': 9
    }

    image_sizes = range(min_size, max_size + size_step, size_step)
    avg_cpu_times = []
    avg_gpu_times = []
    speedups = []

    for size in image_sizes:
        print(f"\nBenchmarking image size: {size}x{size}")
        cpu_times = []
        gpu_times = []
        gpu_compute_only_times = []

        for iteration in range(num_iterations):
            test_image = generate_random_image(size)

            try:
                # CPU computation
                cpu_features, cpu_time = compute_hog_features_cpu(
                    test_image, **hog_params
                )
                cpu_times.append(cpu_time)

                # GPU computation
                gpu_features, gpu_time = compute_hog_features_gpu(
                    test_image, **hog_params
                )
                gpu_times.append(gpu_time)

                # Verify results match (only on first iteration)
                if iteration == 0:
                    if cpu_features.shape != gpu_features.shape:
                        print(f"Warning: Feature dimensions mismatch at size {size}!")
                    else:
                        max_difference = np.abs(cpu_features - gpu_features).max()
                        print(f"Maximum CPU-GPU difference: {max_difference}")

            except cv2.error as e:
                print(f"Error processing size {size}: {e}")
                continue

        # Calculate and store average times
        avg_cpu_time = np.mean(cpu_times)
        avg_gpu_time = np.mean(gpu_times)
        speedup = avg_cpu_time / avg_gpu_time

        avg_cpu_times.append(avg_cpu_time)
        avg_gpu_times.append(avg_gpu_time)
        speedups.append(speedup)

        print(f"Results for {size}x{size}:")
        print(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
        print(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
        print(f"  GPU Speedup: {speedup:.2f}x")

    # Create visualization
    import os
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figure.png')
    visualize_performance_results(list(image_sizes), avg_cpu_times, avg_gpu_times, speedups, script_dir)

    return {
        'sizes': list(image_sizes),
        'cpu_times': avg_cpu_times,
        'gpu_times': avg_gpu_times,
        'speedups': speedups
    }


if __name__ == "__main__":

    print(cv2.getBuildInformation())
    import sys

    print(sys.executable)

    try:
        results = benchmark_hog_performance(
            min_size=128,
            max_size=4096,
            size_step=128,
            num_iterations=5
        )

    except Exception as e:
        print(f"Error during benchmark: {e}")