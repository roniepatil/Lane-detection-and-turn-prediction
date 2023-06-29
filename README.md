# Lane detection and turn prediction
Developed a lane detection algorithm using edge detection, Hough transform, homography, and curve-fitting techniques. Achieved 95% accuracy in predicting car turn direction and computing curvature radius. Used semantic segmentation to identify the drivable areas.

## Author
Rohit M Patil

---
## Results
* **Lane detection**
  -  ![alt text](https://github.com/roniepatil/Lane-detection-and-turn-prediction/blob/main/Images/laneDetection.gif)
* **Turn prediction**
  - ![alt text](https://github.com/roniepatil/Lane-detection-and-turn-prediction/blob/main/Images/turnPrediction.gif)

## Dependencies

| Plugin | 
| ------ |
| scipy | 
| numpy | 
| cv2 | 
| matplotlib | 
| glob | 

## Instructions to run


**Problem 1.a) Histogram equaliztion - Full image:**
```bash
python prob1_all_frames_video.py
```
or
```bash
python3 prob1_all_frames_video.py
```

**Problem 1.b) Adaptive Histogram equaliztion - :**
```bash
python prob1b_all_frames_video.py
```
or
```bash
python3 prob1b_all_frames_video.py
```

**Problem 2) Lane detection:**
```bash
python prob2_video.py
```
or
```bash
python3 prob2_video.py
```


**Problem 3) Turn prediction:**
```bash
python prob3_video.py
```
or
```bash
python3 prob3_video.py
```

**Find all the outputs after running the code in their respective output folder eg:prob1_output, prob2_output, prob3_output **