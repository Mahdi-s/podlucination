import numpy as np
from scipy import signal as sig
import librosa
from tqdm import tqdm
import subprocess
import os
from PIL import Image, ImageDraw, ImageFont
import colorsys

# FFmpeg path configuration
ffmpeg_path = 'ffmpeg'  # Default for Mac/Linux
if os.name == 'nt':  # Windows
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # Replace with your actual FFmpeg path

# Load audio file
audio_path = 'Episode Name.mp3'  # Replace with your audio file path
y, sampleRate = librosa.load(audio_path, sr=None, mono=True)

# Episode and Podcast Information
episode_name = 'Episode 2: How does Multi-Head Attention work?'  # Replace with your episode name
podcast_name = 'Podlucination'     # Replace with your podcast channel name

# Normalize audio data
y = y / np.max(np.abs(y))
y = y * 30

# Set frame rate for video
fps = 30
hop_length = int(sampleRate / fps)
num_frames = int(len(y) / hop_length)

# Video dimensions
width = 1920
height = 1080

# Number of frequency divisions
numDivs = 40

# Pre-compute frequency divisions based on initial PSD
freq, PSD = sig.periodogram(y[:hop_length], sampleRate, nfft=int(sampleRate / 10))

# Generate frequency divisions
freqDivs = []
freqDivs.append([0])
freqDivs.append([1, 2])
for i in range(2, numDivs):
    prevLow = freqDivs[i - 1][0]
    prevHigh = freqDivs[i - 1][1]
    freqDivs.append([prevHigh + 1, prevHigh + 1 + int((prevHigh - prevLow) * 1.0625)])
    if len(freqDivs) == numDivs:
        freqDivs[numDivs - 1] = [freqDivs[numDivs - 1][0], len(freq) - 1]

# Initialize arrays
divs = np.zeros(numDivs)
amp = np.zeros(numDivs)

# Generate vibrant colors with alpha
def get_rainbow_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rgb = tuple(int(x * 255) for x in rgb)
        # Add alpha channel for transparency
        colors.append((rgb[0], rgb[1], rgb[2], 200))
    return colors

colors = get_rainbow_colors(numDivs)

# Particle class for particle system
class Particle:
    def __init__(self, x, y, color):
        self.x = x + np.random.uniform(-5, 5)  # Slight horizontal variation
        self.y = y
        self.vy = np.random.uniform(1, 3)  # Positive for downward movement
        self.size = np.random.uniform(1, 3)  # Smaller particles
        self.alpha = 30  # More transparent particles
        self.color = color

    def update(self):
        self.y += self.vy  # Move downward
        self.alpha -= 2  # Fade out more slowly
        self.size += 0.05  # Slightly grow
        if self.alpha < 0:
            self.alpha = 0

    def is_dead(self):
        return self.alpha <= 0

    def draw(self, draw):
        fill_color = (
            int(self.color[0]),
            int(self.color[1]),
            int(self.color[2]),
            int(self.alpha),
        )
        draw.ellipse(
            [self.x - self.size, self.y - self.size, self.x + self.size, self.y + self.size],
            fill=fill_color,
        )

# Set up ffmpeg writer
writer = subprocess.Popen(
    [
        ffmpeg_path,
        '-y',
        '-f',
        'image2pipe',
        '-vcodec',
        'png',
        '-r',
        str(fps),
        '-i',
        '-',
        '-vcodec',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-r',
        str(fps),
        'video_no_audio.mp4',
    ],
    stdin=subprocess.PIPE,
)

# Visualization scaling factors
SCALE_FACTOR = height / 4
DECAY_FACTOR = 1.8
BAR_WIDTH = 20
CORNER_RADIUS = 10  # Adjust this value to change how rounded the corners are

def draw_rounded_bar(draw, x, height, color, width, center_y):
    # Calculate rectangle coordinates
    left = x - width / 2
    right = x + width / 2
    
    # Ensure positive height by taking absolute value
    height = abs(height)
    
    # Calculate top and bottom ensuring top is always less than bottom
    top = center_y - height
    bottom = center_y + height

    # Ensure top is always less than bottom
    if top > bottom:
        top, bottom = bottom, top

    # Draw a rectangle with rounded corners and transparency
    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=min(CORNER_RADIUS, height/2),  # Ensure radius isn't larger than half the height
        fill=color,
    )

# Function to draw a vertical gradient background
def draw_background_gradient(image, start_color, end_color):
    """Draws a vertical gradient from start_color to end_color on the image."""
    width, height = image.size
    gradient = Image.new('RGBA', (1, height), color=0)
    draw_gradient = ImageDraw.Draw(gradient)
    for y in range(height):
        ratio = y / height
        r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
        g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
        b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
        draw_gradient.point((0, y), (r, g, b, 255))
    gradient = gradient.resize((width, height))
    image.paste(gradient, (0, 0))

# Initialize particle list
particles = []

# Load font
try:
    font_size = 48  # Adjust font size as needed
    # Assuming the font file is in the same directory as the script
    font_path = os.path.join(os.path.dirname(__file__), 'arialroundedmtbold.ttf')
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    # If the font is not found, use a default font
    font = ImageFont.load_default()
    print("Arial Rounded MT Bold font not found. Using default font.")

# Load and prepare profile image
try:
    profile_image_path = os.path.join(os.path.dirname(__file__), 'profile.png')
    original_profile_image = Image.open(profile_image_path).convert("RGBA")
    # Resize profile image to be smaller
    profile_size = int(height * 0.1)  # 10% of the video height
    original_profile_image = original_profile_image.resize((profile_size, profile_size), Image.LANCZOS)
    # Create a circular mask
    mask = Image.new('L', (profile_size, profile_size), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((0, 0, profile_size, profile_size), fill=255)
    original_profile_image.putalpha(mask)
except IOError:
    original_profile_image = None
    print("Profile image not found. Skipping profile picture.")

print('Rendering frames...')
for frame in tqdm(range(num_frames)):
    # Create new image with RGBA mode for transparency
    image = Image.new('RGBA', (width, height))
    # Draw background gradient
    draw_background_gradient(image, (0, 0, 30), (0, 0, 0))  # Dark blue to black
    draw = ImageDraw.Draw(image)

    # Draw episode name
    text = episode_name
    text_color = (255, 255, 255)  # White color
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width) / 2
    text_y = height * 0.05  # Adjust vertical position as needed
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    # Get the audio samples for this frame
    start = frame * hop_length
    end = start + hop_length
    audio_frame = y[start:end]

    if len(audio_frame) < hop_length:
        audio_frame = np.pad(audio_frame, (0, hop_length - len(audio_frame)), 'constant')

    # Compute the periodogram (PSD)
    freq, PSD = sig.periodogram(audio_frame, sampleRate, nfft=int(sampleRate / 10))
    PSD = PSD * 3  # Increase the intensity

    # Calculate base positions
    margin = width * 0.05
    bar_spacing = (width - 2 * margin) / numDivs
    center_y = height / 2

    # Update visualization
    divs[0] = PSD[0] ** 0.1
    bar_height = min(divs[0] * SCALE_FACTOR, height / 2 - 10)
    x = margin
    draw_rounded_bar(draw, x, bar_height, colors[0], BAR_WIDTH, center_y)

    # Generate particles for the first bar
    num_particles = int(bar_height / 10)  # More particles
    for _ in range(num_particles):
        p = Particle(x, center_y + bar_height, colors[0])  # Start at bottom of bar
        particles.append(p)

    for i in range(1, numDivs):
        divs[i] = np.average(PSD[int(freqDivs[i][0]) : int(freqDivs[i][1])]) ** 0.3

        if divs[i] > amp[i]:
            amp[i] = divs[i]
        else:
            amp[i] = max(0, amp[i] - (amp[i] - divs[i]) / DECAY_FACTOR)

        x = margin + i * bar_spacing
        bar_height = min(amp[i] * SCALE_FACTOR, height / 2 - 10)
        draw_rounded_bar(draw, x, bar_height, colors[i], BAR_WIDTH, center_y)

        # Generate particles based on bar height
        num_particles = int(bar_height / 20)  # More particles
        for _ in range(num_particles):
            p = Particle(x, center_y + bar_height, colors[i])  # Start at bottom of bar
            particles.append(p)

    # Update and draw particles
    for p in particles[:]:
        p.update()
        p.draw(draw)
        if p.is_dead():
            particles.remove(p)

    # Paste profile image and podcast name
    if original_profile_image:
        # Calculate rotation angle based on time (one rotation per 30 seconds)
        seconds_per_rotation = 30
        rotation_angle = (frame / fps) * (360 / seconds_per_rotation)  # Convert frame to time and calculate angle
        profile_image = original_profile_image.rotate(rotation_angle, resample=Image.BICUBIC, expand=False)

        profile_x = width - profile_size - int(width * 0.02)  # 2% margin from the right
        profile_y = height - profile_size - int(height * 0.02)  # 2% margin from the bottom

        # Paste profile image
        image.paste(profile_image, (profile_x, profile_y), profile_image)

        # Draw podcast name to the left of the profile picture
        podcast_text = podcast_name
        podcast_font_size = 36  # Adjust font size as needed
        podcast_font = ImageFont.truetype(font_path, podcast_font_size)
        podcast_text_color = (255, 255, 255)  # White color
        podcast_bbox = draw.textbbox((0, 0), podcast_text, font=podcast_font)
        podcast_text_width = podcast_bbox[2] - podcast_bbox[0]
        podcast_text_height = podcast_bbox[3] - podcast_bbox[1]

        # Position the text to the left of the profile picture
        text_margin = int(width * 0.01)  # 1% margin between text and profile picture
        podcast_text_x = profile_x - podcast_text_width - text_margin
        podcast_text_y = profile_y + (profile_size - podcast_text_height) / 2  # Centered vertically with profile picture

        # Ensure the text doesn't go off-screen
        if podcast_text_x < 0:
            podcast_text_x = 0  # Adjust position if needed

        draw.text((podcast_text_x, podcast_text_y), podcast_text, font=podcast_font, fill=podcast_text_color)

    # Convert image to RGB mode before saving
    rgb_image = image.convert('RGB')
    # Save the frame
    rgb_image.save(writer.stdin, 'PNG')

writer.stdin.close()
writer.wait()

# Combine video and audio using ffmpeg
print('Combining audio and video...')
subprocess.run(
    [
        ffmpeg_path,
        '-y',
        '-i',
        'video_no_audio.mp4',
        '-i',
        audio_path,
        '-c:v',
        'copy',
        '-c:a',
        'aac',
        '-b:a',
        '192k',
        '-shortest',
        'output_with_audio.mp4',
    ]
)

# Clean up
os.remove('video_no_audio.mp4')
print('Video saved as output_with_audio.mp4')
