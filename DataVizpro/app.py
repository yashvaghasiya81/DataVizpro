from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import sys

# Add utils folder to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_processing import process_data, perform_eda
from model_training import train_and_evaluate_models
from report_generator import generate_report
from dashboard.dashboard_app import app as dash_app

# Initialize Flask app
app = Flask(__name__)

# Initialize the Dash app with the Flask server
dash_app.init_app(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard/')
def dashboard():
    return dash_app.index()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Process data
        data_summary, heatmap_paths, X_train, X_test, y_train, y_test, target_type = process_data(filepath)

        # Perform EDA
        eda_plots = perform_eda(filepath)

        # Train models with automated selection
        model_results, confusion_matrices, comparison_chart, characteristics = train_and_evaluate_models(
            X_train, X_test, y_train, y_test, target_type
        )

        # Generate report
        report_path = generate_report(
            data_summary, eda_plots, model_results, confusion_matrices, comparison_chart, characteristics
        )

        # Pass data to results page
        return render_template(
            'results.html',
            summary=data_summary,
            eda_plots=eda_plots,
            heatmap=heatmap_paths,
            model_results=model_results,
            confusion_matrices=confusion_matrices,
            comparison_chart=comparison_chart,
            report_path=report_path,
            characteristics=characteristics
        )

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
