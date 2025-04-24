import copy
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from skimage import img_as_float


def get_sliding_windows(image: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a sliding window view of the image for efficient morphological operations.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    kernel_shape : tuple of int
        Shape of the kernel (height, width).

    Returns
    -------
    numpy.ndarray
        Sliding window view of the image.
    """
    k_h, k_w = kernel_shape
    i_h, i_w = image.shape
    stride_h, stride_w = image.strides

    shape = (i_h - k_h + 1, i_w - k_w + 1, k_h, k_w)
    strides = (stride_h, stride_w, stride_h, stride_w)

    return as_strided(image, shape=shape, strides=strides)


def erosion(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform erosion on the input image using the given kernel.

    Parameters
    ----------
    image : numpy.ndarray
        Input binary image.
    kernel : numpy.ndarray
        Structuring element (binary kernel).

    Returns
    -------
    numpy.ndarray
        Eroded binary image.
    """
    k_h, k_w = kernel.shape
    padded_image = np.pad(
        image,
        ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)),
        mode="constant",
        constant_values=0,
    )
    windows = get_sliding_windows(padded_image, kernel.shape)

    return np.all(windows * kernel == kernel, axis=(2, 3)).astype(np.uint8)


def dilation(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform dilation on the input image using the given kernel.

    Parameters
    ----------
    image : numpy.ndarray
        Input binary image.
    kernel : numpy.ndarray
        Structuring element (binary kernel).

    Returns
    -------
    numpy.ndarray
        Dilated binary image.
    """
    k_h, k_w = kernel.shape
    padded_image = np.pad(
        image,
        ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)),
        mode="constant",
        constant_values=0,
    )
    windows = get_sliding_windows(padded_image, kernel.shape)

    return np.any(windows * kernel == 1, axis=(2, 3)).astype(np.uint8)


def opening(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform opening on the input image using the given kernel.

    Opening is defined as erosion followed by dilation.

    Parameters
    ----------
    image : numpy.ndarray
        Input binary image.
    kernel : numpy.ndarray
        Structuring element (binary kernel).

    Returns
    -------
    numpy.ndarray
        Binary image after opening.
    """
    return dilation(erosion(image, kernel), kernel)


def closing(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform closing on the input image using the given kernel.

    Closing is defined as dilation followed by erosion.

    Parameters
    ----------
    image : numpy.ndarray
        Input binary image.
    kernel : numpy.ndarray
        Structuring element (binary kernel).

    Returns
    -------
    numpy.ndarray
        Binary image after closing.
    """
    return erosion(dilation(image, kernel), kernel)


def contrast_filter(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """
    Apply a contrast filter to the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    factor : float, optional
        Contrast adjustment factor. Values > 1 increase contrast, values between 0 and 1 decrease contrast.

    Returns
    -------
    numpy.ndarray
        Image with adjusted contrast.
    """
    mean = np.mean(image)
    adjusted = mean + factor * (image - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Apply a sharpening filter to the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    numpy.ndarray
        Sharpened image.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    k_h, k_w = kernel.shape
    padded_image = np.pad(
        image,
        ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)),
        mode="constant",
        constant_values=0,
    )
    windows = get_sliding_windows(padded_image, kernel.shape)
    sharpened = np.sum(windows * kernel, axis=(2, 3))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def averaging_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply an averaging filter to the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    kernel_size : int, optional
        Size of the averaging kernel (must be odd).

    Returns
    -------
    numpy.ndarray
        Image after applying the averaging filter.
    """
    k_h, k_w = kernel_size, kernel_size
    padded_image = np.pad(
        image,
        ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)),
        mode="constant",
        constant_values=0,
    )
    windows = get_sliding_windows(padded_image, (k_h, k_w))
    averaged = np.mean(windows, axis=(2, 3))
    return averaged.astype(np.uint8)


def sobel_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply the Sobel filter to detect edges in an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    numpy.ndarray
        Edge-detected image.
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    k_h, k_w = sobel_x.shape

    padded_image = np.pad(
        image,
        ((k_h // 2, k_h // 2), (k_w // 2, k_w // 2)),
        mode="constant",
        constant_values=0,
    )
    windows = get_sliding_windows(padded_image, sobel_x.shape)

    grad_x = np.sum(windows * sobel_x, axis=(2, 3))
    grad_y = np.sum(windows * sobel_y, axis=(2, 3))

    edges = np.sqrt(grad_x**2 + grad_y**2)
    if edges.max() > 0:
        edges = (edges / edges.max()) * 255
    else:
        edges = np.zeros_like(edges)

    return edges.astype(np.uint8)


def circular_kernel(radius: int) -> np.ndarray:
    """
    Create a circular kernel with a given radius.

    Parameters
    ----------
    radius : int
        Radius of the circle.

    Returns
    -------
    numpy.ndarray
        Binary circular kernel.
    """
    diameter = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((diameter, diameter), dtype=bool)
    kernel[mask] = 1
    return kernel.astype(np.uint8)


def transform_pupil_img(img: np.ndarray) -> np.ndarray:
    """
    Transform the image of the pupil.

    Parameters
    ----------
    img : numpy.ndarray
        Image of the pupil.

    Returns
    -------
    numpy.ndarray
        Transformed image of the pupil.
    """
    img = contrast_filter(sharpen(img), factor=1.2)
    p_I = np.mean(img) / 6.0
    img = img < p_I

    img = mask_boundary(img, 50)
    img = averaging_filter(img.astype(np.float32), kernel_size=3)
    img = img > 0.1
    img = closing(img, kernel=circular_kernel(9))

    img = averaging_filter(img.astype(np.float32), kernel_size=5)
    img = img > 0.1
    img = closing(img, kernel=circular_kernel(21))

    return img


def transform_pupil_imgs(imgs: np.ndarray) -> np.ndarray:
    """
    Transform pupil images in parallel.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of images to transform.

    Returns
    -------
    numpy.ndarray
        Array of transformed images.
    """
    with ThreadPoolExecutor() as executor:
        transformed_images = list(executor.map(transform_pupil_img, imgs))
    return np.array(transformed_images)


def analyse_pupil_projection(vector: np.ndarray) -> Union[Tuple[int, int], int]:
    """
    Find the maximum non-zero window around the argmax in a 1D vector.

    Parameters
    ----------
    vector : numpy.ndarray
        Input 1D array.

    Returns
    -------
    tuple of int or int
        A tuple (start_index, radius) if non-zero values are found, otherwise 0.
    """
    if np.all(vector == 0):
        return 0

    argmax_idx = np.argmax(vector)

    start_idx = argmax_idx
    while start_idx > 0 and vector[start_idx - 1] > 0:
        start_idx -= 1

    end_idx = argmax_idx
    while end_idx < len(vector) - 1 and vector[end_idx + 1] > 0:
        end_idx += 1

    radius = np.ceil((end_idx - start_idx) / 2).astype(int)
    return start_idx + radius, radius


def get_radial_projection(image: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """
    Compute the radial projection of an image by summing pixel intensities at increasing radii.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale or binary image.
    center : tuple of int
        (x, y) coordinates of the center.

    Returns
    -------
    numpy.ndarray
        1D array representing the sum of pixel intensities at each radius.
    """
    height, width = image.shape
    Y, X = np.ogrid[:height, :width]

    distances = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2).astype(np.int32)
    max_radius = distances.max()

    radial_sum = np.bincount(
        distances.ravel(), weights=image.ravel(), minlength=max_radius + 1
    )

    return radial_sum


def get_pupil(img: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """
    Detect the pupil center and radius based on projection peaks.

    Parameters
    ----------
    img : numpy.ndarray
        Binary or grayscale image.

    Returns
    -------
    tuple
        ((x, y), radius) where (x, y) is the pupil center and radius is the estimated radius.
    """
    if img is None or img.size == 0:
        raise ValueError("Input image is empty or None.")

    vertical_projection = np.sum(img, axis=0)
    horizontal_projection = np.sum(img, axis=1)

    x, radius_x = analyse_pupil_projection(vertical_projection)
    y, radius_y = analyse_pupil_projection(horizontal_projection)
    radius = np.ceil((radius_x + radius_y) / 2).astype(int)

    return (x, y), radius


def get_pupils(imgs: np.ndarray) -> List[Tuple[Tuple[int, int], int]]:
    """
    Detect pupils in parallel.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of images to process.

    Returns
    -------
    list
        List of pupil positions as tuples ((x, y), radius).
    """
    with ThreadPoolExecutor() as executor:
        return list(executor.map(get_pupil, imgs))


def analyse_iris_projection(center: int, pupil_radius: int, vector: np.ndarray) -> int:
    """
    Find the maximum non-zero window around the argmax in a 1D vector for iris detection.

    Uses the property that the iris radius cannot exceed 5 times the pupil radius.

    Parameters
    ----------
    center : int
         Center position in the vector.
    pupil_radius : int
         Radius of the pupil.
    vector : numpy.ndarray
         Input 1D array.

    Returns
    -------
    int
         Maximum distance from the center within the window.
    """
    if np.all(vector == 0):
        return 0

    start_idx = center
    while start_idx > 0 and vector[start_idx - 1] < vector[start_idx]:
        start_idx -= 1

    start_max_idx = start_idx
    while start_idx >= 0 and center - start_idx < 5 * pupil_radius:
        if vector[start_idx] > vector[start_max_idx]:
            start_max_idx = start_idx
        start_idx -= 1

    end_idx = center
    while end_idx < len(vector) - 1 and vector[end_idx + 1] < vector[end_idx]:
        end_idx += 1

    end_max_idx = end_idx
    while end_idx < len(vector) and end_idx - center < 5 * pupil_radius:
        if vector[end_idx] > vector[end_max_idx]:
            end_max_idx = end_idx
        end_idx += 1

    return max(end_max_idx - center, center - start_max_idx)


def mask_boundary(image: np.ndarray, k: int) -> np.ndarray:
    """
    Set pixels within k pixels from the image boundary to 0.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    k : int
        Number of pixels from the boundary to mask.

    Returns
    -------
    numpy.ndarray
        Image with boundary pixels masked.
    """
    mask = np.ones_like(image, dtype=bool)
    mask[:k, :] = 0
    mask[-k:, :] = 0
    mask[:, :k] = 0
    mask[:, -k:] = 0
    return image * mask


def mask_circle(
    image: np.ndarray, center: Tuple[int, int], radius: float, inside: bool = True
) -> np.ndarray:
    """
    Mask pixels inside or outside a given circle.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    center : tuple of int
        (x, y) coordinates of the circle's center.
    radius : float
        Radius of the circle.
    inside : bool, optional
        If True, mask pixels inside the circle; otherwise, mask pixels outside.

    Returns
    -------
    numpy.ndarray
        Masked image.
    """
    height, width = image.shape
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius if inside else dist_from_center > radius
    return image * mask


def transform_iris_img(
    img: np.ndarray, pupil_dt: Tuple[Tuple[int, int], int]
) -> np.ndarray:
    """
    Transform the image of the iris using the pupil detection information.

    Parameters
    ----------
    img : numpy.ndarray
        Image of the iris.
    pupil_dt : tuple
        ([center, pupil_radius]) for the pupil.

    Returns
    -------
    numpy.ndarray
        Transformed image of the iris.
    """
    center, pupil_radius = pupil_dt

    img = sobel_filter(contrast_filter(averaging_filter(img, kernel_size=11)))
    p_I = np.mean(img) / 0.7
    img = img > p_I

    img = mask_boundary(img, 10)
    img = mask_circle(img, center, 5 * pupil_radius)
    img = mask_circle(img, center, 1.8 * pupil_radius, inside=False)

    return img


def transform_iris_imgs(
    imgs: np.ndarray, pupils_dt: List[Tuple[Tuple[int, int], int]]
) -> np.ndarray:
    """
    Transform iris images in parallel using pupil detection information.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of iris images to transform.
    pupils_dt : list
        List of pupil detection information for each image.

    Returns
    -------
    numpy.ndarray
        Array of transformed iris images.
    """
    try:
        with ThreadPoolExecutor() as executor:
            transformed_images = list(
                executor.map(
                    lambda args: transform_iris_img(*args), zip(imgs, pupils_dt)
                )
            )
        return np.array(transformed_images)
    except Exception as e:
        print(f"Error during iris transformation: {e}")
        return np.array([])


def get_iris(
    img: np.ndarray, pupil_dt: Tuple[Tuple[int, int], int]
) -> Tuple[Tuple[int, int], int]:
    """
    Detect the iris center and radius based on projection peaks.

    Parameters
    ----------
    img : numpy.ndarray
        Binary or grayscale iris image.
    pupil_dt : tuple
        [center, pupil_radius] for the pupil.

    Returns
    -------
    tuple
        ((x, y), radius) for the iris.
    """
    (x, y), _ = pupil_dt

    radial_projection = get_radial_projection(img, (x, y))
    return (x, y), np.argmax(radial_projection)


def get_irises(
    imgs: np.ndarray, pupils_dt: List[Tuple[Tuple[int, int], int]]
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Detect irises in parallel.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of iris images.
    pupils_dt : list
        List of pupil detection information.

    Returns
    -------
    list
        List of iris detection results as tuples ((x, y), radius).
    """
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda args: get_iris(*args), zip(imgs, pupils_dt)))


def unwrap_annular_segment(
    image: np.ndarray,
    center: Tuple[int, int],
    radius_1: float,
    radius_2: float,
    theta_range: Tuple[int, int] = (0, 360),
    output_shape: Tuple[int, int] = (50, 150),
) -> np.ndarray:
    """
    Unwrap an annular segment from a circular region into a rectangular image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.
    center : tuple of int
        (x, y) coordinates of the circle's center.
    radius_1 : float
        Inner radius.
    radius_2 : float
        Outer radius.
    theta_range : tuple of int, optional
        (theta_min, theta_max) in degrees.
    output_shape : tuple of int, optional
        (height, width) of the unwrapped image.

    Returns
    -------
    numpy.ndarray
        The unwrapped annular segment as a rectangular image.

    Raises
    ------
    ValueError
        If inner radius is not smaller than outer radius.
    """
    if radius_1 >= radius_2:
        raise ValueError("Inner radius must be smaller than outer radius.")

    x_c, y_c = center
    theta_min, theta_max = np.radians(theta_range)
    height, width = output_shape

    r_values = np.linspace(radius_1, radius_2, height)
    theta_values = np.linspace(theta_min, theta_max, width)

    r_mesh, theta_mesh = np.meshgrid(r_values, theta_values, indexing="ij")

    x_mesh = x_c + r_mesh * np.cos(theta_mesh)
    y_mesh = y_c + r_mesh * np.sin(theta_mesh)

    x_mesh = np.clip(np.round(x_mesh).astype(int), 0, image.shape[1] - 1)
    y_mesh = np.clip(np.round(y_mesh).astype(int), 0, image.shape[0] - 1)

    unwrapped_image = image[y_mesh, x_mesh]

    return unwrapped_image


def get_irises_parts(
    imgs: np.ndarray,
    pupils_dt: List[Tuple[Tuple[int, int], int]],
    iris_dt: List[Tuple[Tuple[int, int], int]],
) -> np.ndarray:
    """
    Extract parts of the iris images in parallel.

    Parameters
    ----------
    imgs : numpy.ndarray
        Array of iris images.
    pupils_dt : list
        List of pupil detection information.
    iris_dt : list
        List of iris detection information.

    Returns
    -------
    numpy.ndarray
        Array of extracted iris part images.
    """
    with ThreadPoolExecutor() as executor:
        transformed_images = list(
            executor.map(
                lambda args: unwrap_annular_segment(
                    args[0],
                    center=args[1][0],
                    radius_1=args[1][1],
                    radius_2=args[2][1],
                ),
                zip(imgs, pupils_dt, iris_dt),
            )
        )
    return np.array(transformed_images)


def save_img(
    img: np.ndarray,
    iris_img: np.ndarray,
    pupil_img: np.ndarray,
    iris_part_img: np.ndarray,
    center: Tuple[int, int],
    iris_radius: float,
    pupil_radius: float,
    label: str,
    images_path: str,
    extracted_irises_path: str,
    irises_path: str,
    pupils_path: str,
) -> None:
    """
    Save annotated images with iris and pupil circles.

    This function saves four images:
      1. The original image with both iris (red) and pupil (green) circles.
      2. The iris image with its circle.
      3. The pupil image with its circle.
      4. The extracted iris part image.

    Parameters
    ----------
    img : numpy.ndarray
        The original grayscale image.
    iris_img : numpy.ndarray
        The iris image.
    pupil_img : numpy.ndarray
        The pupil image.
    iris_part_img : numpy.ndarray
        The extracted iris part image.
    center : tuple of int
        (x, y) coordinates for the center of the circles.
    iris_radius : float
        Radius of the iris circle.
    pupil_radius : float
        Radius of the pupil circle.
    label : str
        Label string in the format "side_idx" (e.g., "left_001").
    images_path : str
        Directory to save the main image.
    extracted_irises_path : str
        Directory to save the extracted iris part image.
    irises_path : str
        Directory to save the iris image.
    pupils_path : str
        Directory to save the pupil image.
    """
    side, idx = label.split("_")

    plt.figure(figsize=(5, 5))
    iris_circle = plt.Circle(center, iris_radius, color="red", fill=False, linewidth=2)
    pupil_circle = plt.Circle(
        center, pupil_radius, color="green", fill=False, linewidth=2
    )
    plt.imshow(img, cmap="gray")
    plt.plot(*center, "ro")
    plt.gca().add_patch(iris_circle)
    plt.gca().add_patch(pupil_circle)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(images_path, side, f"{idx}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    iris_circle = plt.Circle(center, iris_radius, color="red", fill=False, linewidth=2)
    plt.imshow(iris_img, cmap="binary")
    plt.plot(*center, "ro")
    plt.gca().add_patch(iris_circle)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(irises_path, side, f"{idx}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    pupil_circle = plt.Circle(
        center, pupil_radius, color="green", fill=False, linewidth=2
    )
    plt.imshow(pupil_img, cmap="binary")
    plt.plot(*center, "go")
    plt.gca().add_patch(pupil_circle)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(pupils_path, side, f"{idx}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(iris_part_img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(extracted_irises_path, side, f"{idx}.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def create_directories(base_path: str, subdirs: List[str]) -> None:
    """
    Create directories for each subdirectory under the base path.

    Parameters
    ----------
    base_path : str
        The base directory path.
    subdirs : list of str
        List of subdirectory names.
    """
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


def save_imgs(
    imgs: List[np.ndarray],
    iris_imgs: List[np.ndarray],
    pupil_imgs: List[np.ndarray],
    irises_parts: List[np.ndarray],
    irises: List[Tuple[Tuple[int, int], int]],
    pupils: List[Tuple[Any, int]],
    labels: List[str],
) -> None:
    """
    Save multiple annotated images concurrently.

    Parameters
    ----------
    imgs : list of numpy.ndarray
        List of original grayscale images.
    iris_imgs : list of numpy.ndarray
        List of iris images.
    pupil_imgs : list of numpy.ndarray
        List of pupil images.
    irises_parts : list of numpy.ndarray
        List of extracted iris part images.
    irises : list of tuple
        List of tuples (center, iris_radius).
    pupils : list of tuple
        List of tuples (_, pupil_radius).
    labels : list of str
        List of labels formatted as "side_idx" (e.g., "left_001").
    """
    os.makedirs("data", exist_ok=True)

    base_dirs = {
        "images": "data/images",
        "extracted_irises": "data/extracted_irises",
        "irises": "data/irises",
        "pupils": "data/pupils",
    }
    for key, base_path in base_dirs.items():
        create_directories(base_path, ["left", "right"])

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.map(
            lambda args: save_img(
                img=args[0],
                iris_img=args[1],
                pupil_img=args[2],
                iris_part_img=args[3],
                center=args[4][0],
                iris_radius=args[4][1],
                pupil_radius=args[5][1],
                label=args[6],
                images_path=base_dirs["images"],
                extracted_irises_path=base_dirs["extracted_irises"],
                irises_path=base_dirs["irises"],
                pupils_path=base_dirs["pupils"],
            ),
            zip(imgs, iris_imgs, pupil_imgs, irises_parts, irises, pupils, labels),
        )


def downsample_to_128(band: np.ndarray) -> np.ndarray:
    """
    Downsample a 1D signal to 128 columns.

    Parameters
    ----------
    band : numpy.ndarray
        Input 2D array with shape (height, width) or (height, ...).

    Returns
    -------
    numpy.ndarray
        Downsampled array with shape (1, 128).
    """
    h, w = band.shape[:2]
    col_means = band.mean(axis=0)
    group_size = w // 128
    out = col_means.reshape(128, group_size).mean(axis=1)
    return out[np.newaxis, :]


def gabor(band: np.ndarray, frequency: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a Gabor filter to a 1D signal.

    Parameters
    ----------
    band : numpy.ndarray
        Input 1D array (or 2D array with shape (1, N)).
    frequency : float, optional
        Frequency of the sinusoidal component.

    Returns
    -------
    tuple of numpy.ndarray
        (real_resp, imag_resp) with the real and imaginary responses.
    """
    if band.ndim > 1:  # Ensure band is 1D
        band = band.ravel()

    x = np.arange(band.size)
    x_centered = x - band.size // 2
    sigma = 1.0 / frequency

    gauss = np.exp(-0.5 * (x_centered**2) / (sigma**2))
    carrier_real = np.cos(2 * np.pi * frequency * x_centered)
    carrier_imag = np.sin(2 * np.pi * frequency * x_centered)
    kernel_real = gauss * carrier_real
    kernel_imag = gauss * carrier_imag

    kernel_real = kernel_real / np.sqrt(np.sum(kernel_real**2))
    kernel_imag = kernel_imag / np.sqrt(np.sum(kernel_imag**2))
    real_resp = np.convolve(band, kernel_real, mode="same")
    imag_resp = np.convolve(band, kernel_imag, mode="same")

    return real_resp, imag_resp


def image_to_iris_code(
    input_image: np.ndarray,
    number_of_bands: int = 8,
    plot_iris_code: bool = False,
    gabor_frequency: float = 0.25,
) -> np.ndarray:
    """
    Generate an iris code from the input image using Gabor filters.

    Parameters
    ----------
    input_image : numpy.ndarray
        Input image.
    number_of_bands : int, optional
        Number of bands to split the image into.
    plot_iris_code : bool, optional
        If True, plot the generated iris code.
    gabor_frequency : float, optional
        Frequency parameter for the Gabor filter.

    Returns
    -------
    numpy.ndarray
        Iris code represented as a 2D binary array.
    """
    image_temp = copy.deepcopy(input_image)
    rows_to_remove = image_temp.shape[0] - 440
    cropped_image = image_temp[rows_to_remove:, :]

    band_height = cropped_image.shape[0] // number_of_bands
    bands = []
    for i in range(number_of_bands):
        start_row = i * band_height
        end_row = (
            (i + 1) * band_height
            if i != number_of_bands - 1
            else cropped_image.shape[0]
        )
        band = cropped_image[start_row:end_row, :]
        bands.append(band)

    bands_specifications = {
        **{
            str(i): {"top_cutout_width": 130, "bottom_cutout_width": 0}
            for i in range(number_of_bands // 2)
        },
        **{
            str(i): {"top_cutout_width": 322, "bottom_cutout_width": 320}
            for i in range(number_of_bands // 2, number_of_bands // 2 * 3 // 4)
        },
        **{
            str(i): {"top_cutout_width": 386, "bottom_cutout_width": 384}
            for i in range(number_of_bands // 2 * 3 // 4, number_of_bands)
        },
    }

    cropped_bands = []

    for i in range(8):
        top_cutout_width = bands_specifications[str(i)]["top_cutout_width"]
        bottom_cutout_width = bands_specifications[str(i)]["bottom_cutout_width"]

        band = copy.deepcopy(bands[i])
        top_band = copy.deepcopy(band)[:, band.shape[1] // 2 :]
        bottom_band = band[:, : band.shape[1] // 2].copy()

        top_center = top_band.shape[1] // 2
        bottom_center = bottom_band.shape[1] // 2

        if top_cutout_width > 0:
            top_start = max(0, top_center - top_cutout_width // 2)
            top_end = min(top_band.shape[1], top_center + top_cutout_width // 2)
            top_band_cropped = np.delete(top_band, np.s_[top_start:top_end], axis=1)
        else:
            top_band_cropped = top_band

        if bottom_cutout_width > 0:
            bottom_start = max(0, bottom_center - bottom_cutout_width // 2)
            bottom_end = min(
                bottom_band.shape[1], bottom_center + bottom_cutout_width // 2
            )
            bottom_band_cropped = np.delete(
                bottom_band, np.s_[bottom_start:bottom_end], axis=1
            )
        else:
            bottom_band_cropped = bottom_band

        combined_band = np.concatenate((bottom_band_cropped, top_band_cropped), axis=1)
        cropped_bands.append(combined_band.copy())

    cropped_bands_rescaled = np.stack(
        [downsample_to_128(b) for b in cropped_bands], axis=0
    )
    cropped_bands_rescaled = np.array(cropped_bands_rescaled, dtype=float)
    cropped_bands_rescaled = img_as_float(cropped_bands_rescaled)

    iris_codes = []
    frequency = gabor_frequency

    for band in cropped_bands_rescaled:
        real_resp, imag_resp = gabor(band, frequency=frequency)
        real_line = real_resp.ravel()
        imag_line = imag_resp.ravel()

        code = np.zeros((real_line.size, 2), dtype=np.uint8)
        code[:, 0] = (real_line > 0).astype(np.uint8)
        code[:, 1] = (imag_line > 0).astype(np.uint8)

        iris_codes.append(code)

    iris_codes = np.array(iris_codes)
    tmp = iris_codes.transpose(0, 2, 1)
    iris_codes_2row = tmp.reshape(-1, tmp.shape[-1])

    if plot_iris_code:
        plt.figure(figsize=(16, 8))
        plt.imshow(iris_codes_2row, cmap="gray", aspect="auto")
    return iris_codes_2row


def get_cropped_bands_rescaled(
    input_image: np.ndarray,
    plot: bool = False,
    number_of_bands: int = 8,
    plot_iris_code: bool = False,
    gabor_frequency: float = 0.25,
) -> np.ndarray:
    """
    Process and rescale input image bands for iris recognition.

    Parameters
    ----------
    input_image : numpy.ndarray
        The original image array to be processed.
    plot : bool, optional
        Flag to plot intermediate results for debugging. Defaults to False.
    number_of_bands : int, optional
        The total number of bands to split the image into. Defaults to 8.
    plot_iris_code : bool, optional
        Flag indicating whether to plot the iris code. Defaults to False.
    gabor_frequency : float, optional
        Frequency parameter for Gabor filtering. Defaults to 0.25.

    Returns
    -------
    numpy.ndarray
        A numpy array containing the stack of rescaled cropped bands.
    """
    image_temp = copy.deepcopy(input_image)
    rows_to_remove = image_temp.shape[0] - 440
    cropped_image = image_temp[rows_to_remove:, :]

    band_height = cropped_image.shape[0] // number_of_bands
    bands = []
    for i in range(number_of_bands):
        start_row = i * band_height
        end_row = (
            (i + 1) * band_height
            if i != number_of_bands - 1
            else cropped_image.shape[0]
        )
        band = cropped_image[start_row:end_row, :]
        bands.append(band)

    bands_specifications = {
        **{  # Bands 0-3: Top cutout 130, no bottom cutout
            str(i): {"top_cutout_width": 130, "bottom_cutout_width": 0}
            for i in range(number_of_bands // 2)
        },
        **{  # Bands 4-5: Top cutout 322, bottom cutout 320
            str(i): {"top_cutout_width": 322, "bottom_cutout_width": 320}
            for i in range(number_of_bands // 2, number_of_bands // 2 * 3 // 4)
        },
        **{  # Bands 6-7: Top cutout 386, bottom cutout 384
            str(i): {"top_cutout_width": 386, "bottom_cutout_width": 384}
            for i in range(number_of_bands // 2 * 3 // 4, number_of_bands)
        },
    }

    cropped_bands = []

    for i in range(8):
        top_cutout_width = bands_specifications[str(i)]["top_cutout_width"]
        bottom_cutout_width = bands_specifications[str(i)]["bottom_cutout_width"]

        band = copy.deepcopy(bands[i])
        top_band = copy.deepcopy(band)[:, band.shape[1] // 2 :]
        bottom_band = band[:, : band.shape[1] // 2].copy()

        top_center = top_band.shape[1] // 2
        bottom_center = bottom_band.shape[1] // 2

        if top_cutout_width > 0:
            top_start = top_center - top_cutout_width // 2
            top_end = top_center + top_cutout_width // 2
            top_start = max(0, top_start)
            top_end = min(top_band.shape[1], top_end)
            top_band_cropped = np.delete(top_band, np.s_[top_start:top_end], axis=1)
        else:
            top_band_cropped = top_band

        if bottom_cutout_width > 0:
            bottom_start = bottom_center - bottom_cutout_width // 2
            bottom_end = bottom_center + bottom_cutout_width // 2
            bottom_start = max(0, bottom_start)
            bottom_end = min(bottom_band.shape[1], bottom_end)
            bottom_band_cropped = np.delete(
                bottom_band, np.s_[bottom_start:bottom_end], axis=1
            )
        else:
            bottom_band_cropped = bottom_band

        combined_band = np.concatenate((bottom_band_cropped, top_band_cropped), axis=1)
        cropped_bands.append(combined_band.copy())

    cropped_bands_rescaled = np.stack(
        [downsample_to_128(b) for b in cropped_bands], axis=0
    )
    return cropped_bands_rescaled
