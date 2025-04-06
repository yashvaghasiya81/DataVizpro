from fpdf import FPDF
import os

def generate_report(data_summary, eda_plots, model_results, confusion_matrices, model_comparison_chart, characteristics=None):
    report_path = 'static/report.pdf'
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Use built-in fonts
    pdf.add_page()
    pdf.set_font('Helvetica', size=12)

    # Add Title
    pdf.set_font('Helvetica', 'B', size=16)
    pdf.cell(190, 10, 'Data Analysis Report', ln=True, align='C')
    pdf.ln(10)

    # Add Dataset Characteristics if available
    if characteristics:
        pdf.set_font('Helvetica', 'B', size=14)
        pdf.cell(190, 10, 'Dataset Characteristics', ln=True, align='L')
        pdf.set_font('Helvetica', size=12)
        
        # Display characteristics
        characteristics_text = [
            f"Number of Samples: {characteristics['n_samples']}",
            f"Number of Features: {characteristics['n_features']}",
            f"Feature Density: {characteristics['feature_density']:.2f}",
            f"Class Balance: {characteristics['class_balance']:.2f}"
        ]
        
        for text in characteristics_text:
            pdf.cell(190, 10, text, ln=True)
        pdf.ln(5)

        # Add Model Recommendations if available
        if 'recommendations' in model_results[list(model_results.keys())[0]]:
            pdf.set_font('Helvetica', 'B', size=14)
            pdf.cell(190, 10, 'Model Recommendations', ln=True, align='L')
            pdf.set_font('Helvetica', size=12)
            
            for recommendation in model_results[list(model_results.keys())[0]]['recommendations']:
                pdf.multi_cell(190, 10, f"* {recommendation}")
            pdf.ln(5)

    # Add Data Summary
    pdf.set_font('Helvetica', 'B', size=14)
    pdf.cell(190, 10, 'Data Summary', ln=True, align='L')
    pdf.set_font('Helvetica', size=12)

    if data_summary:
        for key, value in data_summary.items():
            pdf.multi_cell(190, 10, f"{key}: {value}")
        pdf.ln(5)
    else:
        pdf.cell(190, 10, 'No data summary available.', ln=True)

    # Add EDA plots
    if eda_plots:
        for plot in eda_plots:
            pdf.add_page()
            try:
                pdf.image(plot, x=10, y=None, w=190)
            except Exception as e:
                pdf.cell(190, 10, f'Error loading plot: {plot}', ln=True)

    # Add Model Results
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', size=14)
    pdf.cell(190, 10, 'Model Results', ln=True, align='L')
    pdf.set_font('Helvetica', size=12)
    
    if model_results:
        for model, report in model_results.items():
            pdf.cell(190, 10, f"Model: {model}", ln=True)
            for metric, score in report.items():
                if metric != 'recommendations':
                    pdf.cell(190, 10, f"{metric}: {score}", ln=True)
            pdf.ln(5)
    else:
        pdf.cell(190, 10, 'No model results available.', ln=True)

    # Add Confusion Matrices
    if confusion_matrices:
        for model, cm_path in confusion_matrices.items():
            pdf.add_page()
            pdf.cell(190, 10, f"Confusion Matrix - {model}", ln=True, align='C')
            try:
                pdf.image(cm_path, x=10, y=None, w=190)
            except Exception as e:
                pdf.cell(190, 10, f'Error loading confusion matrix for {model}', ln=True)

    # Add Model Comparison Chart
    if model_comparison_chart:
        pdf.add_page()
        pdf.cell(190, 10, 'Model Comparison', ln=True, align='C')
        try:
            pdf.image(model_comparison_chart, x=10, y=None, w=190)
        except Exception as e:
            pdf.cell(190, 10, 'Error loading model comparison chart', ln=True)

    # Ensure the static directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Save the PDF
    try:
        pdf.output(report_path)
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        # Try alternative encoding
        try:
            pdf.output(report_path, 'F')
        except Exception as e:
            print(f"Failed to generate PDF with alternative encoding: {str(e)}")
            return None

    return report_path
