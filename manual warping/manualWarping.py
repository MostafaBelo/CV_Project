import cv2 as cv
import numpy as np
import os

# Global variables to store points
points_img1 = []
points_img2 = []
MAX_POINTS = 4 # You can increase this (e.g., to 6 or 8) for RANSAC to work better

# Global variables for images (initialized as None, loaded in main)
img1_original = None
img2_original = None
img1_display = None
img2_display = None

# --- Function to draw lines between corresponding points ---
def draw_matches(img1, points1, img2, points2):
    """Draws lines between corresponding points on a concatenated image."""
    # Ensure both lists have the same number of points for drawing lines
    num_pairs = min(len(points1), len(points2))

    # Create copies to draw on, to avoid modifying the display images directly here
    # if they are passed as img1_display, img2_display which are already copies.
    # Or, ensure img1 and img2 passed are already suitable for drawing.
    # For simplicity, assuming img1 and img2 are the display images ready for drawing.

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create a canvas wide enough to hold both images side-by-side
    # Ensure the canvas has 3 channels if input images do
    num_channels = img1.shape[2] if len(img1.shape) == 3 else 1
    if num_channels == 1: # Grayscale
        vis = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR) # Convert to BGR for colored lines
    else: # Color
        vis = np.zeros((max(h1, h2), w1 + w2, num_channels), dtype=np.uint8)

    # Place images on the canvas
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2 # Corrected slicing for img2

    # Draw lines between corresponding points
    for i in range(num_pairs):
        # Points are already scaled if images were scaled
        p1 = (int(points1[i][0]), int(points1[i][1]))
        p2 = (int(points2[i][0]) + w1, int(points2[i][1])) # Add width of img1 to x-coord for img2
        cv.line(vis, p1, p2, (0, 255, 255), 1) # Cyan color line
        cv.circle(vis, p1, 5, (0, 255, 0), -1) # Green circle on img1 point
        cv.circle(vis, p2, 5, (0, 0, 255), -1) # Red circle on img2 point
        
    # If one list has an extra point (currently being selected), draw it too
    if len(points1) > num_pairs:
        p1 = (int(points1[-1][0]), int(points1[-1][1]))
        cv.circle(vis, p1, 5, (0, 255, 0), -1)
    elif len(points2) > num_pairs:
        p2 = (int(points2[-1][0]) + w1, int(points2[-1][1]))
        cv.circle(vis, p2, 5, (0, 0, 255), -1)

    return vis

# --- Mouse callback function for combined view ---
def select_points_combined_callback(event, x, y, flags, param):
    global points_img1, points_img2, img1_display, img2_display, img1_original, img2_original, MAX_POINTS

    if img1_original is None or img2_original is None: # Images not loaded yet
        return

    h1, w1 = img1_original.shape[:2] # Get width of image 1 to distinguish clicks

    if event == cv.EVENT_LBUTTONDOWN:
        # Determine if click was on image 1 or image 2
        if x < w1: # Click was in the left image area (Image 1)
            if len(points_img1) < MAX_POINTS:
                if len(points_img1) <= len(points_img2): # Allow img1 to lead or be equal
                    points_img1.append((x, y))
                    print(f"Point {len(points_img1)} selected on Image 1: ({x},{y}). Now select corresponding point on Image 2.")
                else:
                    print("Please select the corresponding point on Image 2 first.")
            else:
                print(f"Already selected {MAX_POINTS} points for Image 1. Press 'w' to warp or 'r' to reset.")

        else: # Click was in the right image area (Image 2)
            x2 = x - w1 # Adjust x coordinate to be relative to Image 2's origin
            if len(points_img2) < MAX_POINTS:
                if len(points_img2) < len(points_img1): # Ensure img1 has a point waiting for a pair
                    points_img2.append((x2, y))
                    print(f"Point {len(points_img2)} selected on Image 2: ({x2},{y}).")
                    if len(points_img1) < MAX_POINTS:
                        print(f"Now select point {len(points_img1)+1} on Image 1.")
                else:
                    print("Please select a point on Image 1 first.")
            else:
                print(f"Already selected {MAX_POINTS} points for Image 2. Press 'w' to warp or 'r' to reset.")
        
        # After any click that modifies points, update display
        # Create fresh display copies to draw all current points
        current_img1_display = img1_original.copy()
        current_img2_display = img2_original.copy()

        # Draw all selected points on these fresh copies (not strictly needed if draw_matches handles it)
        # for pt in points_img1:
        #     cv.circle(current_img1_display, pt, 5, (0, 255, 0), -1)
        # for pt in points_img2:
        #     cv.circle(current_img2_display, pt, 5, (0, 0, 255), -1)
            
        # Update the combined display with lines
        combined_display_img = draw_matches(current_img1_display, points_img1, current_img2_display, points_img2)
        cv.imshow("Point Selection (Combined View)", combined_display_img)


# --- Function for Automatic Feature Matching and Warping ---
def automatic_warp(img1_orig, img2_orig, script_dir, max_features=500, good_match_percent=0.15):
    """
    Automatically finds features, matches them, computes homography, and warps img2 to img1.
    """
    if img1_orig is None or img2_orig is None:
        print("Error: Original images not loaded for automatic warping.")
        return

    print("Starting automatic feature matching and warping...")

    # Convert images to grayscale
    img1_gray = cv.cvtColor(img1_orig, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2_orig, cv.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv.ORB_create(nfeatures=max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # --- BEGIN DEBUG PRINTS for descriptors ---
    print(f"[DEBUG] Type of descriptors1: {type(descriptors1)}")
    if hasattr(descriptors1, 'shape'):
        print(f"[DEBUG] Shape of descriptors1: {descriptors1.shape}")
    else:
        print(f"[DEBUG] descriptors1 (no shape): {descriptors1}")

    print(f"[DEBUG] Type of descriptors2: {type(descriptors2)}")
    if hasattr(descriptors2, 'shape'):
        print(f"[DEBUG] Shape of descriptors2: {descriptors2.shape}")
    else:
        print(f"[DEBUG] descriptors2 (no shape): {descriptors2}")
    # --- END DEBUG PRINTS for descriptors ---

    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        print("Error: Could not compute descriptors or no descriptors found (descriptors are None or empty).")
        # Optionally, print more details if they are not None but empty
        if descriptors1 is not None:
            print(f"Length of descriptors1: {len(descriptors1)}")
        if descriptors2 is not None:
            print(f"Length of descriptors2: {len(descriptors2)}")
        return

    # Match features.
    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    
    try:
        matches = matcher.match(descriptors1, descriptors2)
    except cv.error as e:
        print(f"[DEBUG] cv.error during matcher.match: {e}")
        print(f"[DEBUG] descriptors1 shape for match: {descriptors1.shape if hasattr(descriptors1, 'shape') else 'N/A'}")
        print(f"[DEBUG] descriptors2 shape for match: {descriptors2.shape if hasattr(descriptors2, 'shape') else 'N/A'}")
        return # Stop if matching fails

    # --- BEGIN DEBUG PRINTS for matches ---
    print(f"[DEBUG] Type of matches after matcher.match: {type(matches)}")
    if isinstance(matches, tuple):
        print(f"[DEBUG] Matches is a tuple. Content (first few elements): {matches[:5] if len(matches) > 5 else matches}") # Print only a few to avoid long output
        # If it's a tuple and you suspect the list is inside, you could try:
        # if len(matches) > 0 and isinstance(matches[0], list):
        #     print("[DEBUG] First element of tuple is a list, trying to use it.")
        #     matches = matches[0] # THIS IS A GUESS - REMOVE IF NOT APPLICABLE
        # else:
        #     print("[DEBUG] Matches tuple does not seem to contain a list as first element.")
        #     # return # Cannot proceed if matches is an unexpected tuple # Let's try converting instead
    elif isinstance(matches, list):
        print(f"[DEBUG] Matches is a list. Number of matches: {len(matches)}")
        if len(matches) > 0:
            print(f"[DEBUG] Type of first match object: {type(matches[0])}")
    else:
        print(f"[DEBUG] Matches is neither a tuple nor a list. Content: {matches}")
        return # Cannot proceed
    # --- END DEBUG PRINTS for matches ---

    # Ensure matches is a list before sorting
    if isinstance(matches, tuple):
        print("[INFO] Converting matches from tuple to list before sorting.")
        matches = list(matches)
    
    # Now matches should be a list, so sort() will work:
    try:
        matches.sort(key=lambda x: x.distance)
    except AttributeError as e:
        print(f"[DEBUG] AttributeError during sort. Type of matches: {type(matches)}")
        raise e # Re-raise the error after printing debug info
    except Exception as e:
        print(f"[DEBUG] Other error during sort: {e}. Type of matches: {type(matches)}")
        raise e


    # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False) # This is redundant if the above sort works

    # Remove not so good matches
    num_good_matches = int(len(matches) * good_match_percent)
    good_matches = matches[:num_good_matches]

    if len(good_matches) < 4: # Need at least 4 points for homography
        print(f"Not enough good matches found - only {len(good_matches)}. Need at least 4.")
        # Draw top N matches found for debugging
        img_matches_debug = cv.drawMatches(img1_orig, keypoints1, img2_orig, keypoints2, matches[:20], None) # Draw all if few
        cv.imshow("Automatic Matches (Not Enough for Homography)", img_matches_debug)
        return

    # Draw top matches
    img_matches = cv.drawMatches(img1_orig, keypoints1, img2_orig, keypoints2, good_matches, None)
    cv.imshow("Automatic Good Matches", img_matches)
    try:
        cv.imwrite(os.path.join(script_dir, "automatic_matches.png"), img_matches)
        print(f"Automatic matches visualization saved to: {os.path.join(script_dir, 'automatic_matches.png')}")
    except Exception as e:
        print(f"Error saving automatic matches image: {e}")


    # Extract location of good matches
    points1_auto = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2_auto = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1_auto[i, :] = keypoints1[match.queryIdx].pt
        points2_auto[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    homography_matrix_auto, mask_auto = cv.findHomography(points2_auto, points1_auto, cv.RANSAC, 5.0)

    if homography_matrix_auto is not None:
        # Warp image
        height, width, channels = img1_orig.shape
        warped_img2_auto = cv.warpPerspective(img2_orig, homography_matrix_auto, (width, height))
        cv.imshow("Automatically Warped Image 2", warped_img2_auto)

        # --- Stitching (simple overlay) instead of blending ---
        # Create a mask for the warped image to identify non-black regions
        gray_warped = cv.cvtColor(warped_img2_auto, cv.COLOR_BGR2GRAY)
        _, mask_warped = cv.threshold(gray_warped, 1, 255, cv.THRESH_BINARY)
        # cv.imshow("Warped Mask", mask_warped) # Optional: uncomment to view the mask

        # Start with a copy of the first image
        stitched_image_auto = img1_orig.copy()

        # Use the mask to copy pixels from warped_img2_auto to the stitched image.
        # This overlays the valid parts of the warped second image onto the first image.
        # Note: For true panoramic stitching where images extend beyond each other,
        # a more complex approach involving a larger canvas and seam blending is usually needed.
        # This method assumes warped_img2_auto is meant to align within the bounds of img1_orig.
        stitched_image_auto[mask_warped != 0] = warped_img2_auto[mask_warped != 0]
        
        cv.imshow("Automatically Stitched Result", stitched_image_auto)

        # Save results
        auto_warped_path = os.path.join(script_dir, "automatic_warped_output.png")
        auto_stitched_path = os.path.join(script_dir, "automatic_stitched_output.png") # Changed from blended
        try:
            cv.imwrite(auto_warped_path, warped_img2_auto)
            print(f"Automatically warped image saved to: {auto_warped_path}")
            cv.imwrite(auto_stitched_path, stitched_image_auto) # Save stitched image
            print(f"Automatically stitched image saved to: {auto_stitched_path}")
        except Exception as e:
            print(f"Error saving automatically warped/stitched images: {e}")
    else:
        print("Error: Could not calculate homography automatically.")

# --- Main ---
if __name__ == '__main__':
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img1_path = os.path.join(script_dir, "left 2.png")
        img2_path = os.path.join(script_dir, "right 2.png")
        
        img1_original_temp = cv.imread(img1_path)
        img2_original_temp = cv.imread(img2_path)

        if img1_original_temp is None:
            print(f"Error: Could not load Image 1 from {img1_path}")
            exit()
        if img2_original_temp is None:
            print(f"Error: Could not load Image 2 from {img2_path}")
            exit()

        scale_factor = 0.5 
        if scale_factor != 1.0:
            new_width_img1 = int(img1_original_temp.shape[1] * scale_factor)
            new_height_img1 = int(img1_original_temp.shape[0] * scale_factor)
            img1_original = cv.resize(img1_original_temp, (new_width_img1, new_height_img1), interpolation=cv.INTER_AREA)

            new_width_img2 = int(img2_original_temp.shape[1] * scale_factor)
            new_height_img2 = int(img2_original_temp.shape[0] * scale_factor)
            img2_original = cv.resize(img2_original_temp, (new_width_img2, new_height_img2), interpolation=cv.INTER_AREA)
        else:
            img1_original = img1_original_temp
            img2_original = img2_original_temp
            
    except Exception as e:
        print(f"An error occurred loading images: {e}")
        exit()

    img1_display = img1_original.copy()
    img2_display = img2_original.copy()

    cv.namedWindow("Point Selection (Combined View)")
    cv.setMouseCallback("Point Selection (Combined View)", select_points_combined_callback, None)

    print("--- Manual Warping ---")
    print(f"Please select {MAX_POINTS} corresponding points on each image for manual warping.")
    print("Click on 'Image 1 (Left Side)' first for a point, then its corresponding point on 'Image 2 (Right Side)'.")
    print("Repeat until 4 pairs are selected. You will see connecting lines.")
    print("Press 'w' to warp Image 2 to Image 1 using manually selected points.")
    print("Press 'a' to attempt automatic warping.")
    print("Press 'r' to reset manually selected points.")
    print("Press 'q' to quit.")

    # Initial display of the combined image
    combined_display_img = draw_matches(img1_display, points_img1, img2_display, points_img2)
    cv.imshow("Point Selection (Combined View)", combined_display_img)

    while True:
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            points_img1 = []
            points_img2 = []
            img1_display = img1_original.copy() 
            img2_display = img2_original.copy()
            combined_display_img = draw_matches(img1_display, points_img1, img2_display, points_img2)
            cv.imshow("Point Selection (Combined View)", combined_display_img)
            print("Points reset. Select points again for manual warping.")
            print(f"Please select {MAX_POINTS} corresponding points on each image.")
            print("Click on 'Image 1 (Left Side)' first for a point, then its corresponding point on 'Image 2 (Right Side)'.")
            print("Or press 'a' for automatic warping.")
        
        elif key == ord('a'): # Add this new key press for automatic warping
            if img1_original is not None and img2_original is not None:
                automatic_warp(img1_original.copy(), img2_original.copy(), script_dir)
            else:
                print("Images not loaded yet. Cannot perform automatic warp.")

        elif key == ord('w'):
            if len(points_img1) == MAX_POINTS and len(points_img2) == MAX_POINTS:
                print("Calculating homography and warping...")
                pts1_np = np.array(points_img1, dtype=np.float32)
                pts2_np = np.array(points_img2, dtype=np.float32)

                homography_matrix, mask = cv.findHomography(pts2_np, pts1_np, cv.RANSAC, 5.0)
                # Alternatively, for exactly 4 points without RANSAC (less robust):
                # if MAX_POINTS == 4:
                #    homography_matrix = cv.getPerspectiveTransform(pts2_np, pts1_np)

                if homography_matrix is not None:
                    height, width, channels = img1_original.shape
                    warped_img2 = cv.warpPerspective(img2_original, homography_matrix, (width, height))

                    cv.imshow("Warped Image 2 (into Image 1 frame)", warped_img2)
                    print("Warping complete. Displaying 'Warped Image 2 (into Image 1 frame)'.")

                    blended_image = cv.addWeighted(img1_original, 0.5, warped_img2, 0.5, 0)
                    cv.imshow("Blended Result (Image1 + Warped Image2)", blended_image)
                    print("Also displaying a 50/50 blend.")

                    warped_output_path = os.path.join(script_dir, "warped_img2_into_img1_frame.png")
                    blended_output_path = os.path.join(script_dir, "blended_result.png")

                    try:
                        cv.imwrite(warped_output_path, warped_img2)
                        print(f"Warped image saved to: {warped_output_path}")
                        cv.imwrite(blended_output_path, blended_image)
                        print(f"Blended image saved to: {blended_output_path}")
                    except Exception as e:
                        print(f"Error saving images: {e}")
                else:
                    print("Error: Could not calculate homography. Check your points (e.g., ensure they are not collinear) or try different points.")
            else:
                print(f"Please select {MAX_POINTS} points on each image before pressing 'w'.")
                print(f"Currently: Image 1 has {len(points_img1)} points, Image 2 has {len(points_img2)} points.")

    cv.destroyAllWindows()