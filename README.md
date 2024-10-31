# 🎵 Podlucination Audio Visualizer 🎨

Turn your podcast episodes into mesmerizing visual experiences! This script creates stunning frequency-based visualizations for your audio content.

## ✨ Features

- 🌈 Beautiful rainbow frequency bars with smooth animations
- 🎨 Particle effects that dance with your audio
- 🔄 Rotating profile picture integration
- 📝 Episode and podcast name overlay
- 🎯 output in 1080p

## 🛠️ Prerequisites

Make sure you have these installed:
- 🐍 Python 3.x
- 🎬 FFmpeg
- 📦 Required Python packages:
- numpy
- scipy
- librosa
- tqdm
- Pillow

## 🎨 Required Assets

Place these in the same directory as the script:
- 🖼️ `profile.png` - Your podcast logo/profile picture
- 🔤 `arialroundedmtbold.ttf` - replace with your font
- 🎧 Your audio file (MP3 format)

## 🚀 Quick Start

1. Update these variables in the script:

```
python
audio_path = 'Episode Name.mp3' # Your audio file
episode_name = 'Episode 2: How does Multi-Head Attention work?'
podcast_name = 'Podlucination'
```

2. Run the script:

```
bash
python app.py
```

3. ✨ Watch the magic happen! The script will generate:
   - A temporary video without audio
   - The final video with synchronized audio

## 🎥 Output

Your visualization will be saved as `output_with_audio.mp4`, ready to be uploaded to YouTube!

## 🎮 Customization

Feel free to tweak these parameters:
- 🎨 Bar colors and transparency
- 📊 Number of frequency divisions
- ✨ Particle effects
- 🖼️ Background gradient colors
- 📐 Video dimensions

## 🎯 Perfect For

- 🎙️ Podcast episodes
- 🎵 Music visualizations
- 🎬 YouTube content
- 📚 Educational content

## 💡 How It Works

The script analyzes your audio using Fast Fourier Transform (FFT) to break it down into frequency bands. Each band gets its own colorful bar that moves to the beat of your content, creating a dynamic and engaging visual experience!

## 🤝 Contributing

Found a bug or want to improve the visualizer? Feel free to open an issue or submit a pull request!

## 📝 License

Free to use for your content creation needs! Just remember to credit Podlucination if you share the code.

---
Made with 💖 for Podlucination