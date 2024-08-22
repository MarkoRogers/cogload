import cv2
import numpy as np
import csv
import time
import cProfile
import pstats
import io


def calculate_change(current_area, previous_area, threshold=0.05):
    # Compare the current pupil area to the previous area and determine the change level
    if previous_area is None:
        return "No Change"

    change_ratio = abs(current_area - previous_area) / previous_area

    if change_ratio < threshold:
        return "No Change"
    elif change_ratio < 2 * threshold:
        return "Little Change"
    else:
        return "Significant Change"


def process_frame(frame, previous_area):
    # Process a single video frame to extract pupil information
    start_time = time.time()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Threshold the image to create a binary image for contour detection
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    process_time = time.time() - start_time

    pupil_center_x = None
    pupil_center_y = None
    pupil_area = None
    largest_contour = None

    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            # Calculate the center of the largest contour
            pupil_center_x = int(M["m10"] / M["m00"])
            pupil_center_y = int(M["m01"] / M["m00"])

        # Calculate the area of the largest contour
        pupil_area = cv2.contourArea(largest_contour)

        # Determine the change in pupil area from the previous frame
        change = calculate_change(pupil_area, previous_area)
        previous_area = pupil_area

        return {
            'Center_X': pupil_center_x,
            'Center_Y': pupil_center_y,
            'Area': pupil_area,
            'Change': change,
            'Processing_Time': process_time
        }, previous_area, largest_contour, pupil_center_x, pupil_center_y

    return None, previous_area, None, None, None


def process_video(video_path, output_csv_path, frame_skip=2, scaling_factor=1, visualization=False):
    # Open the video file for processing
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("error: could not open video.")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Open CSV file for writing the output data
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Center_X', 'Center_Y', 'Area', 'Change', 'Processing_Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0
        previous_area = None

        # Start profiling the function
        pr = cProfile.Profile()
        pr.enable()

        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame if scaling_factor is not 1
            if scaling_factor != 1:
                frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

            # Process every nth frame as defined by frame_skip
            if frame_count % frame_skip == 0:
                result, previous_area, largest_contour, pupil_center_x, pupil_center_y = process_frame(frame,
                                                                                                       previous_area)

                if result:
                    # Write results to CSV with timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    result['Timestamp'] = timestamp
                    writer.writerow(result)

                    # Optionally visualize the results
                    if visualization and largest_contour is not None:
                        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                        if pupil_center_x is not None and pupil_center_y is not None:
                            cv2.circle(frame, (pupil_center_x, pupil_center_y), 5, (255, 0, 0), -1)
                        cv2.imshow('Pupil Detection', frame)

            frame_count += 1

            if visualization and cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop profiling and release resources
        pr.disable()
        cap.release()
        if visualization:
            cv2.destroyAllWindows()

        # Print profiling results
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


if __name__ == "__main__":
    video_path = 'pupilx.mp4'  # Path to the input video file
    output_csv_path = 'pupil_data.csv'  # Output path for the CSV file
    process_video(video_path, output_csv_path, frame_skip=1, scaling_factor=1, visualization=True)
