
## Video-to-Eye Tracking Pipeline

### Environment
Python 3.15
OpenCV, MediaPipe, NumPy

### Input
C7968.MP4
face_landmarker.task

### Day 2 - Video Frames
python video_reader.py

Outputs:
- results/day2/sample_frames/
- results/day2/frame_timestamps.csv
- results/day2/video_info.txt

### Day 3 - Eye Tracking
python test_eye_video.py --video C7968.MP4 --model face_landmarker.task --output results/day3

Outputs:
- results/day3/annotated.mp4
- results/day3/features.csv
- results/day3/report.txt

### Day 3/4 - Visualization and Feasibility
python eye_visualization.py

Outputs:
- results/day4/face_eye_iris_annotated.mp4
- results/day4/screenshots/
- results/day4/eye_crops/
- results/day4/visual_report.txt

### Current Limitations
Eye and iris landmark availability does not
guarantee accurate anatomical localization.

Experimental pupil segmentation acceptance
was 9.4% on the initial recording.

Pupil boundaries require manual validation.
No lie classification is performed.
