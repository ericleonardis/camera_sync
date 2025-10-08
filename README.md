# Video Synchronization Protocol  
**Title:** Multi-Camera Synchronization Using Python, FFmpeg, and HandBrake  
**Version:** 1.1  
**Python Version:** 3.11.10  
**Author:** Eric Leonardis  
**Last Updated:** 10/8/25  

---

## Overview  

This protocol describes how to synchronize multiple camera recordings based on flash detection. It provides a complete workflow for identifying synchronization flashes, aligning video start times, and producing time-synchronized video clips suitable for further analysis.  

**The procedure includes:**  
- Setting up the Python environment and installing dependencies.  
- Running a synchronization script to detect flash events and compute alignment offsets.  
- Cropping and resampling videos using FFmpeg.  
- (Optional) Automatically generating FFmpeg commands from the alignment data.  

This document is written for undergraduate research assistants and assumes no prior experience with command-line or Python tools.  

Before beginning make sure that you collected video from multiple cameras and you flashed the lights in the room on and off multiple times. Make sure your videos have the full light flashing event visibile from all cameras. 

---

## PART I: Environment Setup  

### 1. Install Miniconda  
Miniconda manages Python versions and packages in isolated environments.  

Go to the Miniconda download page:  
https://docs.anaconda.com/miniconda/  

Download the appropriate installer for your operating system:  

| Operating System | Installer |
|------------------|------------|
| Windows (64-bit) | Miniconda3-latest-Windows-x86_64.exe |
| macOS (Intel) | Miniconda3-latest-MacOSX-x86_64.sh |
| macOS (Apple Silicon) | Miniconda3-latest-MacOSX-arm64.sh |  

Install Miniconda:  

**Windows:** Run the `.exe` installer and check “Add Miniconda to PATH.”  

**macOS:** Run in Terminal:  
```bash
bash Miniconda3-latest-MacOSX-arm64.sh
```
Follow the prompts, then restart Terminal.  

Verify installation:  
```bash
conda --version
```

---

### 2. Create a Dedicated Python Environment  
Create an environment named `video_sync` with Python version 3.11.10:  
```bash
conda create -n video_sync python=3.11.10 -y
```

Activate it:  
```bash
conda activate video_sync
```

Verify:  
```bash
python --version
# Expected: Python 3.11.10
```

---

### 3. Install Required Python Packages  
Install all dependencies using `pip`:  
```bash
pip install opencv-python numpy scipy matplotlib tqdm
```

Verify installation:  
```bash
python -c "import cv2, numpy, scipy, matplotlib, tqdm; print('All packages installed successfully.')"
```

---

### 4. Install FFmpeg  
FFmpeg is required to crop and trim video files.  

**Option A (Recommended):**  
```bash
conda install -c conda-forge ffmpeg -y
```

**Option B (Windows manual install):**  
Download from https://ffmpeg.org/download.html  

Extract to `C:\ffmpeg`.  
Add `C:\ffmpeg\bin` to the Windows PATH under “System Environment Variables.”  

Verify:  
```bash
ffmpeg -version
```

---

### 5. Install HandBrake (Optional)  
HandBrake can be used to visually verify synchronization or compress video files.  

**Windows:** Download from https://handbrake.fr/downloads.php  

**macOS (Homebrew):**  
```bash
brew install handbrake
```

---

### 6. Set Up Project Directory  
Organize your files as follows:  
```
/Users/<username>/projects/video_sync/
├── Cam1_sync.mp4
├── Cam2_sync.mp4
├── Cam3_sync.mp4
├── sync_videos.py
└── output/
```
Keep all unprocessed videos in the main folder and store aligned versions in the `output/` directory.  

---

## PART II: Running the Synchronization Script  

### 1. Purpose  
The Python synchronization script:  
- Measures average brightness per frame across each video.  
- Detects peak flash events.  
- Aligns videos by their first flash.  
- Saves the computed frame shifts for later trimming.  

---

### 2. Synchronization Notebook (`sync_videos.py`)  
Run the synchronization script or notebook cell to process all videos and detect the alignment offsets.  For the notebook we may need to specify the number of flashes that we see in the original video.

---

### 3. Execute the Notebook Cell  

Example output:  
```
Alignment shifts (frames):
Video 1: shift by 0 frames
Video 2: shift by 35 frames
Video 3: shift by 52 frames
Saved frame shifts to camera_shifts.npy
```
The output graph will look like this: 
<img width="1389" height="1389" alt="image" src="https://github.com/user-attachments/assets/25f3a9ba-506c-44a7-b534-29035a993178" />

Make sure after alignment that all of the averaged brightness signals align and there are none that appear out of sync. If so then there was some issue with the peak detection and alignment. 

---

### 4. Convert Frame Shifts to Seconds  
If the recording frame rate is 60 FPS:  

| Video | Shift (frames) | Time offset (seconds) |
|--------|----------------|----------------------|
| Cam1 | 0 | 0.00 |
| Cam2 | 35 | 0.58 |
| Cam3 | 52 | 0.87 |

Conversion formula:  
```
seconds = frames / 60
```

---

### 5. Crop Videos with FFmpeg (Manual)  
Example:  
```bash
ffmpeg -i Cam2_sync.mp4 -ss 00:00:00.58 -t 120 \
  -c:v libx264 -preset superfast -crf 23 -pix_fmt yuv420p \
  -c:a aac -b:a 128k Cam2_aligned.mp4
```

Repeat for each camera using its offset time. The reference camera (earliest flash) uses `-ss 0`.
