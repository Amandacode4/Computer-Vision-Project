The biggest issue with this week's dataset was the insane crowd density—up to 246 people per frame meant constant occlusions. Standard SORT was basically useless since motion models break down when everyone blocks each other. 

To fix this, I decided to build a custom version of ByteTrack from scratch instead of just relying on ultralytics' black box. The main idea was a two-stage matching process: first matching high-confidence boxes, then doing a second IoU pass for the really low-confidence ones (down to 0.01) to rescue tracks that would normally get dropped. 

Since detection recall is the real bottleneck here, I bumped the model up to yolov8x and pushed the resolution to 1920p. It took forever to run on my Mac but it was worth it. I also added a Noise Scaled Adaptive (NSA) Kalman Filter which basically scales the measurement noise based on the YOLO confidence score. Finally, I threw in a quick post-processing script to linearly interpolate missing frames for occluded pedestrians, which really helped boost my IDF1 score.

Given more time, I'd definitely try integrating Re-ID appearance features like DeepSORT or BoTSORT to handle longer occlusions better.
