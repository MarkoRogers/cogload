import cv2
import numpy as np
import csv
import time


def calculate_change(current_area, previous_area, threshold=0.05):
    if previous_area is None:
        return "No Change"

    # Calculate percentage change
    change_ratio = abs(current_area - previous_area) / previous_area

    if change_ratio < threshold:
        return "No Change"
    elif change_ratio < 2 * threshold:
        return "Little Change"
    else:
        return "Significant Change"


def process_video(video_path, output_csv_path, fps=4):
    # Open the video file or capture device
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the interval at which frames should be processed (in seconds)
    frame_interval = int(video_fps / fps)

    # Prepare the CSV file to write the data
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Center_X', 'Center_Y', 'Area', 'Change']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0
        previous_area = None

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame based on the desired fps
            if frame_count % frame_interval == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply GaussianBlur to reduce noise
                blurred = cv2.GaussianBlur(gray, (33, 33), 0)

                # Apply threshold to create a binary image
                _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

                # Find contours in the binary image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Initialize variables to store pupil data
                pupil_center_x = None
                pupil_center_y = None
                pupil_area = None

                # Find the largest contour assuming it's the pupil
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        pupil_center_x = int(M["m10"] / M["m00"])
                        pupil_center_y = int(M["m01"] / M["m00"])
                    pupil_area = cv2.contourArea(largest_contour)

                    # Calculate change in pupil size
                    change = calculate_change(pupil_area, previous_area)
                    previous_area = pupil_area

                    # Get the current timestamp
                    timestamp = time.strftime("%H:%M:%S")

                    # Write to the CSV file
                    writer.writerow({
                        'Timestamp': timestamp,
                        'Center_X': pupil_center_x,
                        'Center_Y': pupil_center_y,
                        'Area': pupil_area,
                        'Change': change
                    })

                    # Draw the contour and center on the image for visualization
                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (pupil_center_x, pupil_center_y), 5, (255, 0, 0), -1)

                # Display the resulting frame
                cv2.imshow('Pupil Detection', frame)

            # Increment the frame count
            frame_count += 1

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Paths
    stime = time.time()
    video_path = 'pupilx.mp4'  # Path to the input video file
    output_csv_path = 'pupil_data.csv'  # Output path for the CSV file

    # Process the video
    process_video(video_path, output_csv_path, fps=4)
    print(time.time()-stime)
