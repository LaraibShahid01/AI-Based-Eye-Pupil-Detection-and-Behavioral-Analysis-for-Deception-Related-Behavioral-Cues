FYP Eye Analysis - Windows

1. Copy C7968.MP4 into this folder.
2. In VS Code open this folder and Terminal > New Terminal.
3. Install: python -m pip install -r requirements.txt
4. Run: python test_eye_video.py --video C7968.MP4 --output results --preview
5. Open results/annotated.mp4, results/features.csv, results/report.txt.

For maximum quality: --max-width 1920 (default). For faster testing: --max-width 1280.
If you press Q in the preview, processing stops early and saves partial results.
The first run downloads the official MediaPipe Face Landmarker model.

NOTE: Measurements are feasibility proxies, not validated pupil diameters,
clinical readings, or proof of deception. The first baseline interval is only
valid if the person was actually answering neutral questions during that time.
