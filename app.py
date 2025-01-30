# app.py
from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class RealESRGANEnhancer:
    def __init__(self, model_name='RealESRGAN_x4plus'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = model_name
        self.initialize_model()

    def initialize_model(self):
        try:
            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            model_urls = {
                'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            }
            
            model_path = load_file_from_url(
                url=model_urls[self.model_name],
                model_dir='weights',
                progress=True,
                file_name=None
            )

            loadnet = torch.load(model_path, weights_only=True)
            if 'params_ema' in loadnet:
                self.model.load_state_dict(loadnet['params_ema'])
            elif 'params' in loadnet:
                self.model.load_state_dict(loadnet['params'])
            
            self.model.eval()
            self.model = self.model.to(self.device)
            
        except Exception as e:
            raise Exception(f"Error initializing model: {str(e)}")

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception(f"Failed to read image: {img_path}")
            
            img = img[:, :, ::-1]
            img = img.astype(np.float32) / 255.
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()
            img = img.unsqueeze(0).to(self.device)
            
            return img
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")

    def postprocess_image(self, tensor):
        try:
            output = tensor.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output, (1, 2, 0))
            output = output[:, :, ::-1] * 255.0
            return output.astype(np.uint8)
            
        except Exception as e:
            raise Exception(f"Error postprocessing image: {str(e)}")

    def enhance_image(self, input_path, output_path):
        try:
            input_img = self.preprocess_image(input_path)
            
            with torch.no_grad():
                output = self.model(input_img)
                output_img = self.postprocess_image(output)
                cv2.imwrite(output_path, output_img)
                
            return True
            
        except Exception as e:
            print(f"Error enhancing image: {str(e)}")
            return False

enhancer = RealESRGANEnhancer()

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/upload')
def upload():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_' + filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'realesrgan_' + filename)
        
        # Check if both input and enhanced images already exist
        if os.path.exists(input_path) and os.path.exists(output_path):
            # If they exist, directly return the results template
            return render_template('results.html',
                                original_image='/static/uploads/input_' + filename,
                                enhanced_image='/static/uploads/realesrgan_' + filename)
        
        # If not, save the new input file and process it
        file.save(input_path)
        success = enhancer.enhance_image(input_path, output_path)
        
        if success:
            return render_template('results.html',
                                original_image='/static/uploads/input_' + filename,
                                enhanced_image='/static/uploads/realesrgan_' + filename)
        else:
            return 'Enhancement failed', 500

@app.route('/enhance-combined', methods=['POST'])
def enhance_combined():
    if 'image' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_' + filename)
        intermediate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'intermediate_' + filename)
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], 'combined_' + filename)
        
        # Save input image
        file.save(input_path)
        
        try:
            # Step 1: Enhance using underwater model
            from model_enhance import enhance_model
            intermediate_path = enhance_model(input_path)
            
            # Step 2: Enhance using RealESRGAN
            success = enhancer.enhance_image(intermediate_path, final_path)
            
            if success:
                return render_template('results.html',
                                    original_image='/static/uploads/input_' + filename,
                                    enhanced_image='/static/uploads/combined_' + filename)
            else:
                return 'Enhancement failed', 500
                
        except Exception as e:
            print(f"Error during enhancement: {str(e)}")
            return 'Enhancement failed', 500

if __name__ == '__main__':
    app.run(debug=True)