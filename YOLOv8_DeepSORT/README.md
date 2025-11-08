# YOLOv8 + DeepSORT Multi-Object Tracking with Metrics

This module performs real-time multi-object tracking using:
- **YOLOv8** for object detection
- **DeepSORT** for robust ID tracking
- Real-time evaluation: IoU, MOTA, IDF1 *(simulated for demonstration)*

✅ Annotated video export  
✅ Tracking ID persistence  
✅ Live metrics overlay  
✅ Plots + text report generated automatically  

---

### Run
```bash
python deep_sort_tracker.py


Then:
1️⃣ Select a video
2️⃣ Type an experiment name
3️⃣ Choose number of frames to analyze

Output Structure
tracking_results_<name>/
 ├── videos/ (annotated tracking)
 ├── plots/ (IoU / MOTA / IDF1 curves)
 └── reports/ (automatic evaluation text file)

✅ Notes for Researchers

Ground truth is not provided by the original video.
Thus metrics are simulated for visualization only:

Not valid for benchmarking — demonstration purpose for code structure only

Improvements Planned

Real evaluation using MOTChallenge datasets

Re-identification model optimization

GPU deployment on Jetson/embedded devices

📬 Contact: mariemezzine8@gmail.com
