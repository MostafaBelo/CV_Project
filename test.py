import cv2
import numpy as np

def to_spherical(img, f):
    """Convert a perspective image to spherical projection."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    y_i, x_i = np.indices((h, w))
    x = (x_i - cx) / f
    y = (y_i - cy) / f
    z = np.ones_like(x)

    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x**2 + z**2))

    x_map = f * theta + cx
    y_map = f * phi + cy

    return cv2.remap(img, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR)

def stitch_images(img1, img2, f):
    # Spherical projection
    sph1 = to_spherical(img1, f)
    sph2 = to_spherical(img2, f)

    # SIFT feature detection
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(sph1, None)
    kp2, des2 = sift.detectAndCompute(sph2, None)

    # Match using FLANN + ratio test
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        raise Exception("Not enough matches found!")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Estimate translation in spherical projection
    shift_x = np.median(pts1[:, 0] - pts2[:, 0])
    shift_y = np.median(pts1[:, 1] - pts2[:, 1])

    # Warp second image
    h, w = sph1.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    warped2 = cv2.warpAffine(sph2, M, (w + int(shift_x), h))

    # Paste first image onto panorama
    result = np.zeros_like(warped2)
    result[:, :w] = sph1

    # Create mask and blend
    mask1 = (result > 0).astype(np.uint8)
    mask2 = (warped2 > 0).astype(np.uint8)
    overlap = cv2.bitwise_and(mask1, mask2)

    alpha = 0.5
    blended = result.copy()
    blended[overlap == 1] = cv2.addWeighted(result, alpha, warped2, 1 - alpha, 0)[overlap == 1]
    blended[mask1 == 0] = warped2[mask1 == 0]

    return blended

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Load images
    img1 = cv2.imread('data/left_3.jpeg')
    img2 = cv2.imread('data/right_3.jpeg')
    if img1 is None or img2 is None:
        raise Exception("Image loading failed. Make sure 'left.jpg' and 'right.jpg' exist.")

    # Focal length estimate
    f = 0.5 * img1.shape[1]

    result = stitch_images(img1, img2, f)

    # Display result
    cv2.namedWindow("Spherical Panorama", cv2.WINDOW_NORMAL)
    cv2.imshow("Spherical Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save result
    cv2.imwrite("spherical_panorama.jpg", result)
