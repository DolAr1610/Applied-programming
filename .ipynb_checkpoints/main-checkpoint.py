# import numpy as np
# from numpy import ndarray
# import matplotlib.pyplot as plt
#
#
# def null_step_5(matrix: np.ndarray) -> np.ndarray:
#     """
#     Set every fifth element of the matrix to zero.
#     """
#     matrix.flat[::5] = 0
#     return matrix
#
#
# def null_step_diagonal(matrix: np.ndarray) -> np.ndarray:
#     """
#     Set the main diagonal of the matrix to zero.
#     """
#     np.fill_diagonal(matrix, 0)
#     return matrix
#
#
# def fill_gold(matrix: np.ndarray, gold_value: float) -> np.ndarray:
#     """
#     Replace all zeros in the matrix with a specified gold_value.
#     """
#     matrix[matrix == 0] = gold_value
#     return matrix
#
#
# def plot_histogram(matrix, title: str):
#     """
#     Display a histogram of the matrix values.
#     """
#     flattened_data = matrix.flatten()
#     plt.hist(flattened_data, bins=50, color='green', edgecolor='black')
#     plt.title(title)
#     plt.xlabel("Значення")
#     plt.ylabel("Кількість")
#     plt.show()
#
#
# def plot_matrix(matrix: np.ndarray):
#     """
#     Display the matrix as an image.
#     """
#     plt.imshow(matrix, cmap='viridis')
#     plt.colorbar()
#     plt.show()
#
#
# def find_median(matrix: np.ndarray) -> ndarray:
#     """
#     Calculate the median value of the matrix.
#     """
#     return np.median(matrix)
#
#
# def find_average(matrix: ndarray) -> ndarray:
#     """
#     Calculate the average (mean) value of the matrix.
#     """
#     return np.mean(matrix)
#
#
# def get_gold_value(mean_value: float, average_value: float) -> float:
#     """
#     Calculate the average of the mean and average values.
#     """
#     return (mean_value + average_value) / 2
#
#
# def find_nearest_value(matrix: ndarray, value: float) -> float:
#     """
#     Find the value in the matrix that is closest to a given value.
#     """
#     diff = np.abs(matrix - value)
#     index = np.unravel_index(diff.argmin(), diff.shape)
#     return matrix[index]
#
#
# def generate_matrix(rows: int, cols: int) -> ndarray:
#     """
#     Generate a random matrix with integer values between 0 and 100.
#     """
#     return np.random.randint(0, 101, (rows, cols))
#
#
# def main():
#     """
#     Main function to execute the matrix operations and display results.
#     """
#     # Generate and display a random matrix.
#     rows, cols = 8, 5
#     matrix = generate_matrix(rows, cols)
#     print(matrix)
#     plot_histogram(matrix, "Original Matrix Histogram")
#
#     # Get user input and display the nearest value in the matrix.
#     value = float(input("Please, enter value to check: "))
#     print("The nearest value in matrix is:", find_nearest_value(matrix, value))
#
#     # Display the average and median values of the matrix.
#     average = find_average(matrix)
#     print(f"The average value of the matrix is: {average}")
#     median = find_median(matrix)
#     print(f"The median value of the matrix is: {median}")
#
#     # Compute the gold_value.
#     gold_value = get_gold_value(median, average)
#
#     # Null every fifth element, fill with gold value, and display.
#     matrix_step_5 = null_step_5(matrix.copy())
#     plot_histogram(matrix_step_5, "Matrix with every 5th element nullified")
#     matrix_step_5 = fill_gold(matrix_step_5, gold_value)
#     plot_histogram(matrix_step_5, "Matrix after filling with gold value")
#
#     # Nullify the diagonal, fill with gold value, and display.
#     matrix_step_diagonal = null_step_diagonal(matrix.copy())
#     plot_histogram(matrix_step_diagonal, "Matrix with diagonal nullified")
#     matrix_step_diagonal = fill_gold(matrix_step_diagonal, gold_value)
#     plot_histogram(matrix_step_diagonal, "Matrix diagonal filled with gold value")
#
#     # Normalize matrix and display.
#     min_value, max_value = matrix.min(), matrix.max()
#     normalized_matrix = (matrix - min_value) / (max_value - min_value)
#     print("\nNormalized matrix:")
#     plot_histogram(normalized_matrix, "Normalized Matrix Histogram")
#
#
# if __name__ == '__main__':
#     main()
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from PIL import Image, ImageEnhance


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from a given filepath and ensure its size is 600x600 or larger.
    """
    img = plt.imread(filepath)
    if img.shape[0] < 600 or img.shape[1] < 600:
        raise ValueError("Зображення повинно мати розмір 600x600 або більше")
    return img


def show_image(image: np.ndarray, title: str = "", cmap: str = None):
    """
    Display an image with an optional title and colormap.
    """
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()


def adjust_contrast(img: Image.Image, level: int) -> Image.Image:
    """
    Adjust the contrast of an image based on the provided level.
    """
    if not 1 <= level <= 10:
        raise ValueError("Рівень контрасту повинен бути від 1 до 10")

    enhancer = ImageEnhance.Contrast(img)
    factor = 1 + (level - 5) * 0.2
    return enhancer.enhance(factor)


def numpy_to_pil(image_np: np.ndarray) -> Image.Image:
    """
    Convert a numpy image array to a PIL Image object.
    """
    return Image.fromarray((image_np * 255).astype(np.uint8))


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a color image to grayscale.
    """
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def split_into_channels(img: np.ndarray):
    """
    Split the image into its RGB channels and display them.
    """
    channels = ["Reds", "Greens", "Blues"]
    plt.figure(figsize=(12, 6))

    for idx, channel in enumerate(channels):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(img[:, :, idx], cmap=channel, vmin=0, vmax=255)
        plt.title(f"{channel[:-1]} Channel")
        plt.axis('off')

    plt.show()


def edge_detection(img_gray: np.ndarray) -> np.ndarray:
    """
    Apply edge detection to a grayscale image.
    """
    # Compute the Sobel gradient for the x and y directions.
    sx = sobel(img_gray, axis=0, mode='constant')
    sy = sobel(img_gray, axis=1, mode='constant')

    # Combine the two gradients to get the edge intensity.
    return np.hypot(sx, sy)


def blur_image(img: np.ndarray, sigma: int = 10) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    """
    return gaussian_filter(img, sigma=(sigma, sigma, 0))


def main():
    """
    Main function to load, process and display various versions of an image.
    """
    image_path = 'School.JPG'
    image = load_image(image_path)

    # Display the original image
    show_image(image, "Оригінальне зображення")

    # Adjust contrast based on user input and display the image
    level = int(input("Enter level of contrast from 1 to 10. 5 - middle"))
    image_pil = numpy_to_pil(image)
    show_image(adjust_contrast(image_pil, level), "З контрастністю")

    # Display individual RGB channels
    split_into_channels(image)

    # Convert to grayscale and display
    show_image(convert_to_grayscale(image), "Чорно-біле зображення", cmap="gray")

    # Blur and display the image
    show_image(blur_image(image), "Розмите зображення")

    # Apply edge detection and display the result
    show_image(edge_detection(convert_to_grayscale(image)), "Edge Detection", cmap="gray")


# Execute the main function when the script is run.
if __name__ == "__main__":
    main()