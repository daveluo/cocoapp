import os
import glob
from flask import Flask
from flask import jsonify
from flask import request, render_template

from cocoapp import app

valid_mimetypes = ['image/jpeg', 'image/png']


def get_predictions(img_name):
    #TODO
    return {
        "bboxes":
        [
            {"x1": 10, "x2": 50, "y1": 10, "y2": 50}
        ],
    }


@app.route('/')
def index():
    # app.logger.warning('sample message')
    # # Sort files by upload date
    # recent_files = sorted(
    #     glob.glob("%s/*" % app.config['UPLOAD_FOLDER']),
    #     key=os.path.getctime, reverse=True
    # )
    # # Pick the most recent two or less for the index view
    # slice_index = 2 if len(recent_files) > 1 else len(recent_files)
    # recents = recent_files[:slice_index]
    # return render_template('index.html', recents=recents)

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'no file'}), 400
        # Image info
        img_file = request.files.get('file')
        img_name = img_file.filename
        mimetype = img_file.content_type
        # Return an error if not a valid mimetype
        if mimetype not in valid_mimetypes:
            return jsonify({'error': 'bad-type'})
        # Write image to static directory
        img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        # Run Prediction on the model
        results = get_predictions(img_name)

        # Delete image when done with analysis
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], img_name))

        return jsonify(results)
