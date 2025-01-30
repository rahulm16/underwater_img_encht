# Underwater Image Enhancement Web Application

A Flask-based web application that enhances underwater images using a combination of specialized models. This project provides a user-friendly interface for uploading underwater images and offers two enhancement options:
1. RealESRGAN enhancement for general super-resolution and detail enhancement
2. Combined enhancement using both underwater-specific and RealESRGAN models for optimal results

## Features

- 🌊 Specialized underwater image enhancement
- 🤖 Dual AI-powered processing:
  - RealESRGAN model for super-resolution
  - Custom underwater enhancement model for color correction
- 🎨 Two processing options:
  - Quick enhancement using RealESRGAN
  - Advanced enhancement using combined models
- 🖼️ Support for various image formats
- ⚡ Real-time processing
- 📱 Responsive web design
- 💾 Automatic result saving
- 🔄 Batch processing capability

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **Deep Learning**: 
  - PyTorch (RealESRGAN)
  - TensorFlow (Underwater Enhancement Model)
- **Image Processing**: OpenCV, Pillow
- **Styling**: Font Awesome icons

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/underwater-image-enhancement.git
cd underwater-image-enhancement
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p static/uploads
mkdir -p weights
```

## Usage

1. Start the Flask application:
```bash
python a.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Click the "Try it Now" button on the landing page
   - Upload an underwater image using drag-and-drop or file selection
   - Choose an enhancement option (Quick or Advanced)
   - Click "Enhance Image" to process the image
   - Download the enhanced result

## Project Structure

```
underwater_img_encht
├── static/
│   ├── uploads/         # Stores uploaded and processed images
│   └── underwater-bg.jpg
├── templates/
│   ├── index.html      # Upload page
│   ├── landing.html    # Home page
│   └── results.html    # Results display page
├── a.py               # Main Flask application
├── readme.md
└── requirements.txt
```

## API Endpoints

- `GET /` - Landing page
- `GET /upload` - Image upload page
- `POST /enhance` - Image enhancement endpoint

## Model Details

The application uses the RealESRGAN model for image enhancement:
- Architecture: RRDBNet
- Scale: 4x
- Features: 64 channels
- Blocks: 23 RRDB blocks

Additionally, it uses a custom underwater enhancement model for color correction:
- Architecture: U-Net
- Features: 32 channels
- Layers: 5

## Performance

- Supports images up to 16MB
- Processing time varies based on image size and server capabilities
- GPU acceleration available if CUDA is detected

## Development

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Basic understanding of Flask, PyTorch, and TensorFlow

### Local Development

1. Enable debug mode in Flask:
```python
app.run(debug=True)
```

2. Make changes to the code
3. The Flask development server will automatically reload

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Authors

- Bhavana G
- Bhavana Kabbur

## License

This project is licensed under the MIT License.

## Acknowledgments

- RealESRGAN team for the base model
- Flask team for the web framework
- Open source community for various dependencies

## Troubleshooting

### Common Issues

1. **Image Upload Fails**
   - Check image size (max 16MB)
   - Verify supported format
   - Ensure upload directory exists

2. **Processing Errors**
   - Check GPU memory availability
   - Verify model weights downloaded correctly
   - Check system memory usage

### Getting Help

- Open an issue on GitHub
- Check existing documentation
- Contact the development team

## Future Improvements

- [ ] Add batch processing capability
- [ ] Implement user accounts and image history
- [ ] Add more enhancement models
- [ ] Optimize processing speed
- [ ] Add API documentation
- [ ] Implement progress tracking
