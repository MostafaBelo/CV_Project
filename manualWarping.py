import cv2
import numpy as np
import os

def stitch_images(image_paths, output_path="stitched_panorama.jpg"):
    """
    Stitches a list of images together to create a panorama.

    Args:
        image_paths (list): A list of paths to the images to be stitched.
        output_path (str): Path to save the stitched panorama.

    Returns:
        numpy.ndarray: The stitched image if successful, None otherwise.
    """
    images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Error: Image path '{image_path}' does not exist.")
            return None
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from '{image_path}'.")
            return None
        images.append(img)

    if len(images) < 2:
        print("Error: At least two images are required for stitching.")
        return None

    print(f"Attempting to stitch {len(images)} images...")

    # Create a Stitcher object
    # cv2.Stitcher_PANORAMA is a good general-purpose mode.
    # It tries to determine the best warping strategy (planar, cylindrical, spherical).
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except AttributeError:
        # For older OpenCV versions (e.g., some 4.x versions might use this)
        try:
            stitcher = cv2.createStitcher(False) # Or cv2.createStitcher()
        except AttributeError:
            print("Error: cv2.Stitcher_create() or cv2.createStitcher() not found.")
            print("Please ensure you have opencv-contrib-python installed and a compatible OpenCV version.")
            return None


    # Perform stitching
    (status, stitched_image) = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("Stitching successful!")
        try:
            # Display the stitched image
            cv2.imshow("Stitched Panorama", stitched_image)
            # Save the stitched image
            cv2.imwrite(output_path, stitched_image)
            print(f"Stitched panorama saved to {output_path}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return stitched_image
        except Exception as e:
            print(f"Error displaying/saving image: {e}")
            return None
            
    elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
        print("Stitching failed: Not enough images to stitch or not enough keypoints.")
    elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
        print("Stitching failed: Homography estimation failed.")
        print("This often means not enough matching features or insufficient overlap.")
    elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
        print("Stitching failed: Camera parameters adjustment failed.")
    else:
        print(f"Stitching failed with error code: {status}")
        
    return None

if __name__ == "__main__":
    # --- IMPORTANT: Replace with the actual paths to your images ---
    # For your soccer field, you'll likely have two images
    left_image_path = "left_2.jpeg"  # e.g., "soccer_left.jpg"
    right_image_path = "right_2.jpeg" # e.g., "soccer_right.jpg"

    # Create dummy images if they don't exist for testing purposes
    # You should replace this with your actual image files
    for p in [left_image_path, right_image_path]:
        if not os.path.exists(p):
            print(f"Creating dummy image: {p} (replace with your actual image)")
            dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)
            if "left" in p:
                cv2.putText(dummy_img, "LEFT IMAGE", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            else:
                cv2.putText(dummy_img, "RIGHT IMAGE", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 3)
            cv2.imwrite(p, dummy_img)


    image_list = [left_image_path, right_image_path]
    
    # You can also try stitching more than two images if you have them
    # image_list = ["image1.jpg", "image2.jpg", "image3.jpg"]

    stitched_result = stitch_images(image_list, "stitched_soccer_field.jpg")

    if stitched_result is not None:
        print("Process completed.")
    else:
        print("Stitching process could not be completed successfully.")