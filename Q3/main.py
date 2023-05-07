import time
import cv2
import numpy as np

if __name__ == "__main__":
    # remove unnecessary imports if run externally
    import timeit
    from matplotlib import pyplot as plt
    from tqdm import tqdm


def filter2D(image, kernel):
    """
    Applies a 2D convolution filter to the input image using the provided kernel.

    Args:
        image: numpy.ndarray - Input image to apply the filter on.
        kernel: numpy.ndarray - 2D kernel filter to be applied on the image.

    Returns:
        output: numpy.ndarray - Filtered image after applying the kernel.
    """
    # Get the dimensions of the input image and the kernel.
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the input image to ensure that the output image is the same size as the input.
    pad_height = int(kernel_height / 2)
    pad_width = int(kernel_width / 2)
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')

    # Initialize the output image.
    output = np.zeros((image_height, image_width), dtype=np.float64)

    # Perform the 2D convolution on the padded image using the kernel.
    for i in range(pad_height, image_height + pad_height):
        for j in range(pad_width, image_width + pad_width):
            region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            output[i - pad_height, j - pad_width] = np.sum(np.multiply(region, kernel))

    np.clip(output, 0, 255, out=output)

    return output.astype(np.uint8)


def smooth_and_detect_edges(
        image_filename: str,
        filter_width: int = 3,
        filter_type: str = "gaussian",
        std_dev: float = None,
        t1: int = None,
        th: int = None) -> (np.ndarray, np.ndarray):
    """
    This function takes an input grayscale image and applies a noise-filtering filter kernel of a given width and type.
    The filtered image is then used to calculate edges and then hysteresis thresholding is applied to enhance the edges
    in the image. Finally, the output image is displayed.

    Args:
    - image_filename: A string representing the filename of the input image to be processed.
    - filter_width: An integer representing the width of the filter kernel.
    - filter_type: A string representing the type of noise filtering to be applied.
                   Allowed values: "gaussian", "median", "mean"
    - std_dev: A float representing the standard deviation used to create the Gaussian filter kernel.
               This argument is optional and only required if filter_type is "gaussian".
    - t1: An integer representing the low threshold value used for hysteresis thresholding.
          This argument is optional and only required if hysteresis thresholding is to be applied.
    - th: An integer representing the high threshold value used for hysteresis thresholding.
          This argument is optional and only required if hysteresis thresholding is to be applied.

    Returns:
    - edges: A numpy array representing the edges detected in the input image.
    - img: A numpy array representing the input image.
    """

    # Read input image and convert to grayscale
    _img = cv2.imread(image_filename)
    gray_img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    # Create filter kernel and apply it
    if filter_type == "gaussian":
        if std_dev is None:
            std_dev = 1.5 * filter_width
        kernel = cv2.getGaussianKernel(filter_width, std_dev)
        smoothed_img = filter2D(gray_img, kernel)

    elif filter_type == "median":
        smoothed_img = cv2.medianBlur(gray_img, filter_width)

    elif filter_type == "mean":
        kernel = np.ones((filter_width, filter_width), np.float32) / (filter_width ** 2)
        smoothed_img = filter2D(gray_img, kernel)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Define sobel filter kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Apply sobel filter
    grad_x = filter2D(smoothed_img, sobel_x)
    grad_y = filter2D(smoothed_img, sobel_y)

    grad_mag = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    # Set default values for hysteresis thresholding
    if t1 is None or th is None:
        t1 = 100
        th = 200
    # Apply hysteresis thresholding
    edges = cv2.inRange(grad_mag, t1, th)

    # Display results
    return edges, _img


if __name__ == "__main__":
    my_version, img = smooth_and_detect_edges("lena_bw.png", 3, "gaussian")
    CV2_version = cv2.Canny(cv2.imread("lena_bw.png", cv2.IMREAD_GRAYSCALE), 100, 200)

    # Display results
    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(my_version, cmap='gray')
    plt.title('My attempt'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(CV2_version, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    # performance testing
    print("\tPerformance testing")
    runs = 10
    start = time.time()
    for _ in tqdm(range(runs), desc="Benchmarking smooth_and_detect_edges"):
        smooth_and_detect_edges("lena_bw.png", 3, "gaussian")
    print(f"Finished my smooth_and_detect_edges in\n\t{(time.time() - start) / runs:.4f} secs on average")

    run_time = timeit.timeit(stmt=lambda: cv2.Canny(cv2.imread("lena_bw.png", cv2.IMREAD_GRAYSCALE), 100, 200),
                             number=runs)
    print(f"Finished CV2 canny in\n\t{run_time / runs:.4f} seconds on average")
