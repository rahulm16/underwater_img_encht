# Underwater Image Enhancement Using Deep Learning

## Project Overview
This project implements a deep learning-based solution for enhancing underwater images. It provides a web interface where users can upload underwater images and receive enhanced versions with improved visibility, color, and clarity. The system uses advanced neural network architectures to correct color distortion, remove haze, and improve overall image quality.

## Features
- User-friendly web interface
- Real-time image preview before enhancement
- Support for various image formats
- Instant enhancement using deep learning
- Side-by-side comparison of original and enhanced images
- Automatic color correction and contrast enhancement
- Haze removal and clarity improvement
- Batch processing capabilities

## Technology Stack
- **Frontend**: HTML, CSS, Bootstrap, JavaScript
- **Backend**: Flask (Python)
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: PIL, Matplotlib
- **Other**: NumPy, Pandas

## Technical Details

### Model Architecture
The deep learning model uses a modified U-Net architecture with:
- Encoder-decoder network structure
- Skip connections for feature preservation
- Custom loss functions for color and contrast enhancement
- Attention mechanisms for focusing on degraded areas
- ResNet-based backbone for feature extraction

### Image Processing Capabilities
- Color correction and white balance adjustment
- Contrast enhancement and histogram equalization
- Dehazing and turbidity removal
- Noise reduction and detail preservation
- Brightness adjustment and tone mapping

### Performance Metrics
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Color accuracy measurement
- Processing time per image: ~2-3 seconds
- Support for images up to 4K resolution

## Project Structure
```
Underwater_img_encht/
│
├── static/
│   ├── css/
│   ├── images/
│   └── upload&result/
│
├── templates/
│   ├── enhance.html
│   ├── index.html
│   ├── layout.html
│   ├── result.html
│   └── upload.html
│
├── app.py
├── model_enhance.py
├── underwater_enhancement_model.h5
├── requirements.txt
└── README.md
```

## Installation & Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd Underwater_img_encht
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Access the application at `http://localhost:5000`

## Usage
1. Navigate to the home page
2. Click on "Enhancement" in the navigation bar
3. Upload an underwater image
4. Click "Enhance" to process the image
5. View the enhanced result alongside the original image

## Model Details
The project uses a deep learning model trained specifically for underwater image enhancement. The model architecture is based on a convolutional neural network that learns to improve:
- Color balance
- Contrast
- Clarity
- Visibility
- Overall image quality

## System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Minimum 8GB RAM
- 100MB disk space for the model
- Supported OS: Windows 10/11, Linux, macOS

## Best Practices
- Use high-resolution input images for better results
- Ensure proper lighting conditions in original images
- Avoid heavily compressed input images
- Recommended image formats: PNG, TIFF, high-quality JPEG
- Maximum file size: 10MB per image

## Future Enhancements
- Real-time video enhancement
- Mobile application support
- Batch processing interface
- API integration capabilities
- Custom model training options

## Troubleshooting
Common issues and solutions:
- Image upload errors: Check file format and size
- Processing timeout: Reduce image resolution
- GPU memory errors: Adjust batch size
- Model loading issues: Verify model file integrity

## Contributors
- Bhavana G
- Bhavana Kabbur

## License
Copyright © 2024. All rights reserved.

## Notes
- Ensure you have sufficient system resources for running the deep learning model
- Supported image formats: JPG, PNG
- Maximum image size: 10MB