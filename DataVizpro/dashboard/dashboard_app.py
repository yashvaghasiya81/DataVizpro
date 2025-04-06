import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import base64
import io
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from utils.data_processing import process_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff

# Initialize the Dash app with Flask server=False and a modern theme
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.FLATLY,
                    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
                    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
                ],
                server=False,
                url_base_pathname='/dashboard/')

# Enhanced Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>DataViz Pro - Advanced Data Analysis</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-color: #4A90E2;
                --secondary-color: #2C3E50;
                --accent-color: #16A085;
                --background-color: #F5F8FA;
                --card-bg: #FFFFFF;
                --success-color: #2ECC71;
                --warning-color: #F1C40F;
                --danger-color: #E74C3C;
            }
            
            body {
                background-color: var(--background-color);
                font-family: 'Roboto', sans-serif;
            }
            
            /* Modern Navigation Styling */
            .navbar {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .navbar-brand {
                color: white !important;
                font-weight: 500;
                font-size: 1.3rem;
            }
            
            .nav-link {
                color: rgba(255,255,255,0.9) !important;
                transition: all 0.3s ease;
                position: relative;
            }
            
            .nav-link:hover {
                color: white !important;
                transform: translateY(-2px);
            }
            
            .nav-link::after {
                content: '';
                position: absolute;
                width: 0;
                height: 2px;
                bottom: 0;
                left: 0;
                background-color: white;
                transition: width 0.3s ease;
            }
            
            .nav-link:hover::after {
                width: 100%;
            }
            
            /* Card Styling */
            .dashboard-card {
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
                margin-bottom: 25px;
                background-color: var(--card-bg);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: none;
            }
            
            .dashboard-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            }
            
            .card-header {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white;
                padding: 15px 20px;
                border-radius: 15px 15px 0 0;
                font-weight: 500;
                font-size: 1.1rem;
                border: none;
            }
            
            /* Upload Box Styling */
            .upload-box {
                border: 2px dashed var(--primary-color);
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                background-color: rgba(74, 144, 226, 0.05);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .upload-box:hover {
                border-color: var(--accent-color);
                background-color: rgba(22, 160, 133, 0.05);
                transform: scale(1.02);
            }
            
            /* Dropdown Styling */
            .feature-dropdown .Select-control {
                border-radius: 10px;
                border: 1px solid #E0E7FF;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }
            
            .feature-dropdown .Select-control:hover {
                border-color: var(--primary-color);
                box-shadow: 0 4px 6px rgba(74, 144, 226, 0.1);
            }
            
            /* Button Styling */
            .btn {
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: 500;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .btn-success {
                background: linear-gradient(135deg, var(--success-color), var(--accent-color));
                border: none;
                box-shadow: 0 4px 6px rgba(46, 204, 113, 0.2);
            }
            
            .btn-success:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 8px rgba(46, 204, 113, 0.3);
            }
            
            /* Graph Styling */
            .js-plotly-plot {
                border-radius: 10px;
                padding: 15px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }
            
            .js-plotly-plot:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            /* Tab Styling */
            .nav-tabs {
                border-bottom: none;
                margin-bottom: 20px;
            }
            
            .nav-tabs .nav-link {
                border: none;
                color: var(--secondary-color) !important;
                border-radius: 10px;
                padding: 10px 20px;
                margin-right: 10px;
                transition: all 0.3s ease;
            }
            
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white !important;
                box-shadow: 0 4px 6px rgba(74, 144, 226, 0.2);
            }
            
            /* Progress Bar Styling */
            .progress {
                height: 10px;
                border-radius: 5px;
                background-color: #E0E7FF;
            }
            
            .progress-bar {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                border-radius: 5px;
            }
            
            /* Alert Styling */
            .alert {
                border-radius: 10px;
                border: none;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }
            
            /* Data Table Styling */
            .data-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                margin-top: 10px;
                border-radius: 10px;
                overflow: hidden;
            }
            
            .data-table th {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white;
                padding: 15px;
                font-weight: 500;
            }
            
            .data-table td {
                padding: 12px;
                border-bottom: 1px solid #E0E7FF;
            }
            
            .data-table tr:nth-child(even) {
                background-color: rgba(74, 144, 226, 0.05);
            }
            
            .data-table tr:hover {
                background-color: rgba(22, 160, 133, 0.05);
            }
            
            /* Badge Styling */
            .badge {
                padding: 8px 12px;
                border-radius: 8px;
                font-weight: 500;
                margin: 0 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            /* Animation Classes */
            .fade-in {
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .dashboard-card {
                    margin-bottom: 15px;
                }
                
                .card-header {
                    padding: 12px 15px;
                }
                
                .upload-box {
                    padding: 20px;
                }
            }
            
            /* Visualization Container Styles */
            .viz-container {
                position: relative;
                width: 100%;
                height: 500px;
                overflow: hidden;
                background: white;
                border-radius: 10px;
                transition: all 0.3s ease;
            }
            
            .viz-container:hover {
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .viz-container .js-plotly-plot {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                padding: 15px;
            }
            
            /* Tab Content Styles */
            .tab-content {
                position: relative;
                min-height: 600px;
                background: white;
                border-radius: 0 0 10px 10px;
                overflow: hidden;
            }
            
            .tab-pane {
                position: relative;
                height: 100%;
            }
            
            /* Custom Tabs Styles */
            .custom-tabs .nav-tabs {
                border-bottom: 2px solid #E0E7FF;
                padding: 0 15px;
            }
            
            .custom-tabs .nav-link {
                border: none;
                color: var(--secondary-color);
                padding: 12px 20px;
                margin-right: 5px;
                border-radius: 8px 8px 0 0;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            
            .custom-tabs .nav-link:hover {
                background-color: rgba(74, 144, 226, 0.1);
                color: var(--primary-color);
            }
            
            .custom-tabs .nav-link.active {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white;
                box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
            }
            
            /* Responsive Adjustments */
            @media (max-width: 768px) {
                .viz-container {
                    height: 400px;
                }
                
                .tab-content {
                    min-height: 500px;
                }
                
                .custom-tabs .nav-link {
                    padding: 8px 15px;
                    font-size: 0.9rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Define the layout with improved UI
app.layout = dbc.Container([
    # Data Store
    dcc.Store(id='stored-data', storage_type='memory'),
    
    # Navigation Bar
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-brain me-2"),  # AI/ML themed icon
                "DataViz Pro"
            ], className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-home me-1"),
                    "Home"
                ], href="/")),
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-book me-1"),
                    "Documentation"
                ], href="#")),
                dbc.NavItem(dbc.NavLink([
                    html.I(className="fas fa-info-circle me-1"),
                    "About"
                ], href="#")),
            ], className="ms-auto", navbar=True),
        ]), 
        className="mb-4",
        dark=True,
        color="transparent"
    ),

    # Main Content
    dbc.Row([
        dbc.Col([
            # Upload Section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-robot me-2"),  # AI-themed icon
                    " Upload Your Data"
                ], className="card-header"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            html.I(className="fas fa-brain fa-3x mb-3 text-primary"),  # AI brain icon
                            html.Br(),
                            html.H5("Drag and Drop or Click to Upload", className="mb-2"),
                            html.P([
                                "Upload your dataset ",
                                html.A("(CSV format)", className="text-primary")
                            ], className="text-muted")
                        ]),
                        className="upload-box fade-in",
                        multiple=False
                    ),
                ])
            ], className="dashboard-card")
        ], width=12)
    ]),

    # Data Summary and Analysis Section
    dbc.Row([
        # Data Summary Card
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-microchip me-2"),  # AI-themed icon
                    " Data Insights"
                ], className="card-header"),
                dbc.CardBody(
                    html.Div(id='data-summary', className="p-3")
                )
            ], className="dashboard-card")
        ], width=6),

        # Analysis Configuration Card
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-network-wired me-2"),  # AI network icon
                    " Model Configuration"
                ], className="card-header"),
                dbc.CardBody([
                    # Target Variable Selection with modern styling
                    html.Label([
                        html.I(className="fas fa-bullseye me-2"),
                        "Target Variable"
                    ], className="mb-2 d-flex align-items-center"),
                    dcc.Dropdown(
                        id='target-selector',
                        placeholder="Select target variable...",
                        className="feature-dropdown mb-3"
                    ),
                    
                    # Feature Selection
                    html.Label([
                        html.I(className="fas fa-code-branch me-2"),
                        "Features"
                    ], className="mb-2 d-flex align-items-center"),
                    dcc.Dropdown(
                        id='feature-selector',
                        multi=True,
                        placeholder="Select features for analysis...",
                        className="feature-dropdown mb-3"
                    ),
                    
                    # Model Selection with AI icons
                    html.Label([
                        html.I(className="fas fa-brain me-2"),
                        "Model Type"
                    ], className="mb-2 d-flex align-items-center"),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[
                            {'label': '🤖 Linear Regression', 'value': 'linear'},
                            {'label': '🧠 Logistic Regression', 'value': 'logistic'},
                            {'label': '🌳 Random Forest', 'value': 'rf'},
                            {'label': '🎯 Support Vector Machine', 'value': 'svm'},
                            {'label': '👥 K-Nearest Neighbors', 'value': 'knn'},
                            {'label': '🌲 Decision Tree', 'value': 'dt'},
                            {'label': '📈 Gradient Boosting', 'value': 'gb'},
                            {'label': '⚡ AdaBoost', 'value': 'ada'},
                            {'label': '📊 Naive Bayes', 'value': 'nb'},
                            {'label': '🧠 Neural Network', 'value': 'nn'}
                        ],
                        placeholder="Select model type...",
                        className="feature-dropdown mb-3"
                    ),
                    
                    # Visualization Type
                    html.Label([
                        html.I(className="fas fa-chart-network me-2"),
                        "Visualization"
                    ], className="mb-2 d-flex align-items-center"),
                    dcc.Dropdown(
                        id='viz-type',
                        options=[
                            {'label': '📊 Interactive Scatter', 'value': 'scatter'},
                            {'label': '📈 Smart Histogram', 'value': 'histogram'},
                            {'label': '📦 Dynamic Box Plot', 'value': 'box'},
                            {'label': '🔥 Correlation Matrix', 'value': 'heatmap'},
                            {'label': '🎻 Violin Plot', 'value': 'violin'},
                            {'label': '�� Scatter Matrix', 'value': 'scatter_matrix'},
                            {'label': '📉 Line Plot', 'value': 'line'},
                            {'label': '📊 Bar Chart', 'value': 'bar'},
                            {'label': '🎯 Bubble Chart', 'value': 'bubble'},
                            {'label': '🌡️ Heatmap with Clustering', 'value': 'clustered_heatmap'}
                        ],
                        value='scatter',
                        className="feature-dropdown mb-3"
                    ),
                    
                    # Train Model Button with loading animation
                    dbc.Button([
                        html.I(className="fas fa-cogs me-2"),
                        "Train Model",
                        html.Span(className="ms-2 spinner-border spinner-border-sm d-none", id="training-spinner")
                    ],
                    id="train-model-button",
                    color="success",
                    className="w-100 mt-3 d-flex align-items-center justify-content-center"
                    ),
                ])
            ], className="dashboard-card")
        ], width=6)
    ]),

    # Model Results Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-robot me-2"),  # AI robot icon
                    " Model Performance Analytics"
                ], className="card-header"),
                dbc.CardBody(
                    html.Div(id='model-results', className="p-3")
                )
            ], className="dashboard-card")
        ], width=12)
    ]),

    # Model Comparison Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-balance-scale me-2"),
                    " Model Comparison"
                ], className="card-header"),
                dbc.CardBody([
                    # Model Selection for Comparison
                    dbc.Row([
                        dbc.Col([
                            html.Label([
                                html.I(className="fas fa-cogs me-2"),
                                "Select Models to Compare"
                            ], className="mb-2"),
                            dcc.Dropdown(
                                id='model-comparison-selector',
                                multi=True,
                                options=[
                                    {'label': '🤖 Linear Regression', 'value': 'linear'},
                                    {'label': '🧠 Logistic Regression', 'value': 'logistic'},
                                    {'label': '🌳 Random Forest', 'value': 'rf'},
                                    {'label': '🎯 Support Vector Machine', 'value': 'svm'},
                                    {'label': '👥 K-Nearest Neighbors', 'value': 'knn'},
                                    {'label': '🌲 Decision Tree', 'value': 'dt'},
                                    {'label': '📈 Gradient Boosting', 'value': 'gb'},
                                    {'label': '⚡ AdaBoost', 'value': 'ada'},
                                    {'label': '📊 Naive Bayes', 'value': 'nb'},
                                    {'label': '🧠 Neural Network', 'value': 'nn'}
                                ],
                                placeholder="Select models to compare...",
                                className="feature-dropdown mb-3"
                            )
                        ], width=12)
                    ]),
                    
                    # Comparison Results
                    html.Div(id='model-comparison-results', className="mt-4")
                ])
            ], className="dashboard-card")
        ], width=12)
    ]),

    # Visualization Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-project-diagram me-2"),
                    " Advanced Visualizations"
                ], className="card-header"),
                dbc.CardBody([
                    # Modern tabs with fixed height container
                    dbc.Tabs([
                        # Distribution Analysis Tab
                        dbc.Tab([
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-chart-area me-2"),
                                    "Distribution Analysis"
                                ], className="mt-3 mb-3"),
                                dcc.Dropdown(
                                    id='dist-feature-selector',
                                    placeholder="Select feature for distribution analysis...",
                                    className="feature-dropdown mb-3"
                                ),
                                html.Div([
                                    dcc.Graph(
                                        id='distribution-plot',
                                        config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                        className="border rounded shadow-sm",
                                        style={'height': '500px', 'width': '100%'}
                                    )
                                ], className="viz-container")
                            ], className="p-3")
                        ], label="Distribution", tab_id="tab-dist", label_style={"cursor": "pointer"}),
                        
                        # Relationship Analysis Tab
                        dbc.Tab([
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-bezier-curve me-2"),
                                    "Relationship Analysis"
                                ], className="mt-3 mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='relationship-x-selector',
                                            placeholder="Select X variable...",
                                            className="feature-dropdown mb-3"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='relationship-y-selector',
                                            placeholder="Select Y variable...",
                                            className="feature-dropdown mb-3"
                                        )
                                    ], width=6)
                                ]),
                                html.Div([
                                    dcc.Graph(
                                        id='relationship-plot',
                                        config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                        className="border rounded shadow-sm",
                                        style={'height': '500px', 'width': '100%'}
                                    )
                                ], className="viz-container")
                            ], className="p-3")
                        ], label="Relationships", tab_id="tab-rel", label_style={"cursor": "pointer"}),
                        
                        # Correlation Analysis Tab
                        dbc.Tab([
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-network-wired me-2"),
                                    "Correlation Analysis"
                                ], className="mt-3 mb-3"),
                                dcc.Dropdown(
                                    id='correlation-features-selector',
                                    multi=True,
                                    placeholder="Select features for correlation analysis...",
                                    className="feature-dropdown mb-3"
                                ),
                                html.Div([
                                    dcc.Graph(
                                        id='correlation-plot',
                                        config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                        className="border rounded shadow-sm",
                                        style={'height': '500px', 'width': '100%'}
                                    )
                                ], className="viz-container")
                            ], className="p-3")
                        ], label="Correlations", tab_id="tab-corr", label_style={"cursor": "pointer"}),
                        
                        # Time Series Analysis Tab
                        dbc.Tab([
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-wave-square me-2"),
                                    "Time Series Analysis"
                                ], className="mt-3 mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='time-feature-selector',
                                            placeholder="Select time variable...",
                                            className="feature-dropdown mb-3"
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dcc.Dropdown(
                                            id='value-feature-selector',
                                            placeholder="Select value variable...",
                                            className="feature-dropdown mb-3"
                                        )
                                    ], width=6)
                                ]),
                                html.Div([
                                    dcc.Graph(
                                        id='time-series-plot',
                                        config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                        className="border rounded shadow-sm",
                                        style={'height': '500px', 'width': '100%'}
                                    )
                                ], className="viz-container")
                            ], className="p-3")
                        ], label="Time Series", tab_id="tab-time", label_style={"cursor": "pointer"}),
                    ], id="viz-tabs", active_tab="tab-dist", className="custom-tabs")
                ])
            ], className="dashboard-card", style={'height': '100%'})
        ], width=12)
    ]),

    # Data Table Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-database me-2"),  # Database icon
                    " Data Preview"
                ], className="card-header"),
                dbc.CardBody(
                    html.Div(id='data-table', className="table-responsive")
                )
            ], className="dashboard-card")
        ], width=12)
    ])
], fluid=True, className="px-4 py-3")

# Callbacks for interactivity
@app.callback(
    [Output('data-summary', 'children'),
     Output('feature-selector', 'options'),
     Output('target-selector', 'options'),
     Output('data-table', 'children'),
     Output('stored-data', 'data')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_data(contents, filename):
    if contents is None:
        return (
            html.Div([
                html.I(className="fas fa-robot fa-3x text-primary mb-3"),
                html.H4("Ready to Analyze", className="mb-3"),
                html.P("Upload your dataset to begin the analysis", className="text-muted")
            ], className="text-center py-4 fade-in"),
            [], [], 
            html.Div([
                html.I(className="fas fa-database fa-3x text-muted mb-3"),
                html.P("No data available", className="text-muted")
            ], className="text-center py-4"),
            None
        )
    
    try:
        # Process the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Try different encodings with error handling
        encodings = [
            'utf-8', 
            'utf-8-sig',  # UTF-8 with BOM
            'latin1', 
            'iso-8859-1', 
            'cp1252',
            'ascii',
            'utf-16',
            'utf-32',
            'big5',
            'gb18030'
        ]
        
        df = None
        error_messages = []
        
        # First try to detect encoding
        try:
            import chardet
            detected = chardet.detect(decoded)
            if detected and detected['encoding']:
                encodings.insert(0, detected['encoding'])
        except ImportError:
            pass
        
        for encoding in encodings:
            try:
                # Try reading with current encoding
                if encoding in ['utf-16', 'utf-32']:
                    # For UTF-16/32, use StringIO with decoded text
                    text = decoded.decode(encoding)
                    df = pd.read_csv(io.StringIO(text))
                else:
                    # For other encodings, use BytesIO with raw bytes
                    df = pd.read_csv(io.BytesIO(decoded), encoding=encoding)
                
                # If successful, break the loop
                print(f"Successfully read file with {encoding} encoding")
                break
                
            except UnicodeDecodeError as e:
                error_messages.append(f"Failed to decode with {encoding}: {str(e)}")
                continue
            except pd.errors.ParserError as e:
                error_messages.append(f"CSV parsing error with {encoding}: {str(e)}")
                continue
            except Exception as e:
                error_messages.append(f"Unexpected error with {encoding}: {str(e)}")
                continue
        
        if df is None:
            raise Exception(f"Could not read file with any encoding.\nAttempted encodings: {', '.join(encodings)}\nErrors:\n" + "\n".join(error_messages))
        
        # Clean column names - remove non-ASCII characters and spaces
        df.columns = df.columns.str.encode('ascii', 'ignore').str.decode('ascii')
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Basic data cleaning
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values for each column based on data type
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # For numeric columns, fill with median
                df[column] = df[column].fillna(df[column].median())
            else:
                # For other types, fill with mode
                df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else "Unknown")
        
        # Convert problematic data types
        for column in df.columns:
            try:
                # Try to convert to numeric if possible
                pd.to_numeric(df[column], errors='raise')
            except:
                # If not numeric, ensure string type
                df[column] = df[column].astype(str)
        
        # Store the data
        stored_data = df.to_json(date_format='iso', orient='split')
        
        # Create options for both feature selector and target selector
        column_options = [{'label': col, 'value': col} for col in df.columns]
        
        # Create enhanced data summary with AI/ML theme
        summary = html.Div([
            dbc.Row([
                # Dataset Overview
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-microchip me-2"),
                            "Dataset Overview"
                        ], className="mb-3"),
                        dbc.Card([
                            dbc.ListGroup([
                                dbc.ListGroupItem([
                                    html.I(className="fas fa-table me-2 text-primary"),
                                    html.Strong("Rows: "),
                                    f"{len(df):,}"
                                ], className="d-flex align-items-center"),
                                dbc.ListGroupItem([
                                    html.I(className="fas fa-columns me-2 text-primary"),
                                    html.Strong("Columns: "),
                                    f"{len(df.columns):,}"
                                ], className="d-flex align-items-center"),
                                dbc.ListGroupItem([
                                    html.I(className="fas fa-memory me-2 text-primary"),
                                    html.Strong("Memory Usage: "),
                                    f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                                ], className="d-flex align-items-center")
                            ], flush=True)
                        ], className="border-0 shadow-sm")
                    ], className="mb-4")
                ], width=12),
                
                # Data Types Summary
                dbc.Col([
                    html.H5([
                        html.I(className="fas fa-code-branch me-2"),
                        "Data Types"
                    ], className="mb-3"),
                    html.Div([
                        dbc.Badge(
                            [
                                html.I(className="fas fa-database me-1"),
                                f"{col}: {str(dtype)}"
                            ],
                            color="primary" if "float" in str(dtype) else
                                  "success" if "int" in str(dtype) else
                                  "warning" if "object" in str(dtype) else
                                  "info",
                            className="me-2 mb-2 p-2"
                        ) for col, dtype in df.dtypes.items()
                    ])
                ], width=12, className="mb-4"),
                
                # Quick Statistics for Numeric Columns
                dbc.Col([
                    html.H5([
                        html.I(className="fas fa-chart-bar me-2"),
                        "Quick Statistics"
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6(col, className="card-subtitle mb-2 text-muted"),
                                    html.P([
                                        html.Strong("Mean: "),
                                        f"{df[col].mean():.2f}"
                                    ], className="mb-1") if df[col].dtype in ['int64', 'float64'] else None,
                                    html.P([
                                        html.Strong("Std: "),
                                        f"{df[col].std():.2f}"
                                    ], className="mb-0") if df[col].dtype in ['int64', 'float64'] else None,
                                    html.P([
                                        html.Strong("Unique Values: "),
                                        f"{df[col].nunique()}"
                                    ], className="mb-0") if df[col].dtype not in ['int64', 'float64'] else None,
                                    dbc.Progress(
                                        value=50,
                                        color="info",
                                        className="mt-2",
                                        style={"height": "4px"}
                                    )
                                ])
                            ], className="shadow-sm mb-3")
                        ], width=6) for col in df.columns[:6]  # Limit to first 6 columns
                    ])
                ], width=12)
            ])
        ], className="fade-in")
        
        # Create enhanced data table
        table = html.Div([
            dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th(col, className="text-center") for col in df.columns
                    ]), className="table-dark"
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            str(df.iloc[i][col])[:100] + '...' if len(str(df.iloc[i][col])) > 100 else str(df.iloc[i][col]),
                            className="text-center",
                            style={'background-color': 'rgba(74, 144, 226, 0.05)'} if i % 2 else {}
                        ) for col in df.columns
                    ]) for i in range(min(5, len(df)))
                ])
            ], className="table table-hover table-bordered", style={'borderRadius': '10px'})
        ], className="table-responsive fade-in")
        
        return summary, column_options, column_options, table, stored_data
        
    except Exception as e:
        error_message = str(e)
        suggestions = [
            "Ensure your file is a valid CSV format",
            "Check if the file is not corrupted",
            "Try saving the file with UTF-8 encoding",
            "Remove any special characters from column names",
            "Make sure the file is not empty",
            "Try opening the file in a text editor and saving it with UTF-8 encoding",
            "If using Excel, try 'Save As' and choose CSV UTF-8 format",
            "Check for and remove any non-printable characters"
        ]
        
        return (
            html.Div([
                dbc.Alert([
                    html.H4([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Error Processing File"
                    ], className="alert-heading mb-3"),
                    html.Hr(),
                    html.P([
                        html.Strong("Error Details: "),
                        error_message
                    ], className="mb-3"),
                    html.Hr(),
                    html.H6([
                        html.I(className="fas fa-lightbulb me-2"),
                        "Suggestions:"
                    ], className="mb-2"),
                    html.Ul([
                        html.Li(suggestion) for suggestion in suggestions
                    ], className="mb-0")
                ], color="danger", className="mb-0 fade-in")
            ], className="py-4"),
            [], [], 
            html.Div([
                html.I(className="fas fa-times-circle fa-3x text-danger mb-3"),
                html.P("Error loading data", className="text-danger")
            ], className="text-center py-4 fade-in"),
            None
        )

@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('feature-selector', 'value'),
     Input('viz-type', 'value'),
     Input('stored-data', 'data')]
)
def update_plot(selected_features, viz_type, stored_data):
    if not selected_features or stored_data is None:
        return create_empty_figure("Select features to begin visualization")
    
    # Load the stored data
    df = pd.read_json(stored_data, orient='split')
    
    # Color scheme
    colors = {
        'primary': '#18BC9C',
        'secondary': '#2C3E50',
        'accent': '#E74C3C',
        'light': '#ECF0F1'
    }
    
    # Create figure with custom styling
    if viz_type == 'scatter':
        if len(selected_features) >= 2:
            fig = px.scatter(
                df, 
                x=selected_features[0], 
                y=selected_features[1],
                template='plotly_white',
                title=f'Scatter Plot: {selected_features[0]} vs {selected_features[1]}',
                color_discrete_sequence=[colors['primary']],
                opacity=0.7,
                marginal_x='histogram',
                marginal_y='histogram',
                trendline='ols'
            )
            fig.update_traces(
                marker=dict(
                    size=10,
                    line=dict(width=1, color=colors['secondary'])
                )
            )
        else:
            fig = go.Figure()
            
    elif viz_type == 'histogram':
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[selected_features[0]],
            name='Histogram',
            nbinsx=30,
            marker_color=colors['primary'],
            opacity=0.7
        ))
        
        # Add KDE
        kde = gaussian_kde(df[selected_features[0]].dropna())
        x_range = np.linspace(df[selected_features[0]].min(), df[selected_features[0]].max(), 100)
        kde_vals = kde(x_range)
        
        hist_vals, _ = np.histogram(df[selected_features[0]].dropna(), bins=30)
        scaling_factor = hist_vals.max() / kde_vals.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_vals * scaling_factor,
            name='Density',
            line=dict(color=colors['accent'], width=2),
            mode='lines'
        ))
        
    elif viz_type == 'box':
        fig = px.box(
            df, 
            y=selected_features[0],
            template='plotly_white',
            title=f'Box Plot: {selected_features[0]}',
            points='all',
            notched=True,
            color_discrete_sequence=[colors['primary']]
        )
        
        fig.add_trace(go.Violin(
            y=df[selected_features[0]],
            name='Distribution',
            side='right',
            line_color=colors['accent'],
            fillcolor='rgba(231, 76, 60, 0.1)',
            opacity=0.3
        ))
        
    elif viz_type == 'heatmap':
        if len(selected_features) > 1:
            corr_matrix = df[selected_features].corr()
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=corr_matrix,
                x=selected_features,
                y=selected_features,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
    elif viz_type == 'violin':
        fig = go.Figure()
        for feature in selected_features:
            fig.add_trace(go.Violin(
                y=df[feature],
                name=feature,
                box_visible=True,
                meanline_visible=True,
                points='outliers',
                line_color=colors['primary'],
                fillcolor='rgba(24, 188, 156, 0.2)'
            ))
        fig.update_layout(
            title='Violin Plots',
            showlegend=True
        )
        
    elif viz_type == 'scatter_matrix':
        if len(selected_features) > 1:
            fig = px.scatter_matrix(
                df,
                dimensions=selected_features,
                color=selected_features[0],
                title='Scatter Matrix',
                template='plotly_white'
            )
            fig.update_traces(
                diagonal_visible=False,
                showupperhalf=False,
                marker=dict(
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color=colors['secondary'])
                )
            )
        else:
            fig = go.Figure()
            
    elif viz_type == 'line':
        fig = go.Figure()
        for feature in selected_features:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[feature],
                name=feature,
                mode='lines+markers',
                line=dict(color=colors['primary'], width=2),
                marker=dict(size=6)
            ))
        fig.update_layout(
            title='Line Plot',
            xaxis_title='Index',
            showlegend=True
        )
        
    elif viz_type == 'bar':
        fig = go.Figure()
        for feature in selected_features:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[feature],
                name=feature,
                marker_color=colors['primary'],
                opacity=0.7
            ))
        fig.update_layout(
            title='Bar Chart',
            xaxis_title='Index',
            showlegend=True,
            barmode='group'
        )
        
    elif viz_type == 'bubble':
        if len(selected_features) >= 3:
            fig = px.scatter(
                df,
                x=selected_features[0],
                y=selected_features[1],
                size=selected_features[2],
                color=selected_features[0],
                template='plotly_white',
                title='Bubble Chart',
                size_max=60
            )
        else:
            fig = go.Figure()
            
    elif viz_type == 'clustered_heatmap':
        if len(selected_features) > 1:
            # Calculate correlation matrix
            corr_matrix = df[selected_features].corr()
            
            # Perform hierarchical clustering
            from scipy.cluster import hierarchy
            linkage = hierarchy.linkage(corr_matrix, method='average')
            order = hierarchy.dendrogram(linkage, no_plot=True)['leaves']
            
            # Reorder correlation matrix
            corr_matrix = corr_matrix.iloc[order, order]
            
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            # Add dendrogram
            fig.add_trace(go.Scatter(
                x=np.arange(len(order)),
                y=np.zeros(len(order)),
                mode='markers',
                marker=dict(size=0),
                showlegend=False
            ))
            
    else:
        fig = go.Figure()
    
    # Apply common layout settings
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={
            'family': 'Arial',
            'size': 12,
            'color': colors['secondary']
        },
        title={
            'font': {
                'size': 24,
                'color': colors['secondary']
            },
            'x': 0.5,
            'xanchor': 'center'
        },
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=colors['secondary'],
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['secondary'],
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['secondary'],
            tickfont=dict(size=12)
        )
    )
    
    return fig

# Update the model training callback
@app.callback(
    Output('model-results', 'children'),
    [Input('train-model-button', 'n_clicks')],
    [State('stored-data', 'data'),
     State('target-selector', 'value'),
     State('feature-selector', 'value'),
     State('model-selector', 'value')]
)
def train_model(n_clicks, stored_data, target, features, model_type):
    if n_clicks is None or not stored_data or not target or not features or not model_type:
        return html.Div([
            html.I(className="fas fa-robot fa-3x text-primary mb-3"),
            html.H4("AI Model Ready", className="mb-3"),
            html.P([
                "Configure your model parameters above and click ",
                html.Strong("Train Model"),
                " to begin analysis"
            ], className="text-muted")
        ], className="text-center py-4 fade-in")
    
    try:
        # Load and prepare data
        df = pd.read_json(stored_data, orient='split')
        X = df[features].copy()
        y = df[target].copy()
        
        # Function to convert datetime to numeric
        def convert_datetime(series):
            if pd.api.types.is_datetime64_any_dtype(series):
                return pd.to_numeric(series.astype(np.int64))
            return series

        # Handle different data types in features
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = convert_datetime(X[col])
            elif pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
                X[col] = pd.Categorical(X[col]).codes
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].mean())
        
        # Handle target variable
        if pd.api.types.is_datetime64_any_dtype(y):
            y = convert_datetime(y)
            is_categorical = False
            class_names = None
        else:
            # Check if target is categorical
            unique_values = pd.unique(y)
            is_categorical = (y.dtype == 'object' or 
                            pd.api.types.is_categorical_dtype(y) or 
                            (len(unique_values) < 10 and not np.issubdtype(y.dtype, np.floating)))
            
            if is_categorical:
                # Handle missing values in target
                y = y.fillna(y.mode()[0])
                # Store original class names before encoding
                class_names = np.array([str(val) for val in unique_values])
                # Encode categorical target
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                # Convert to numeric and handle missing values
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.mean())
                class_names = None
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model based on target type and selected model
        if is_categorical:
            if model_type == 'linear':
                model = LogisticRegression(max_iter=1000, random_state=42)
            elif model_type == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', probability=True, random_state=42)
            elif model_type == 'knn':
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_type == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            elif model_type == 'gb':
                model = GradientBoostingClassifier(random_state=42)
            elif model_type == 'ada':
                model = AdaBoostClassifier(random_state=42)
            elif model_type == 'nb':
                model = GaussianNB()
            elif model_type == 'nn':
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            metric_name = 'Accuracy'
        else:
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'rf':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'svm':
                model = SVR(kernel='rbf')
            elif model_type == 'knn':
                model = KNeighborsRegressor(n_neighbors=5)
            elif model_type == 'dt':
                model = DecisionTreeRegressor(random_state=42)
            elif model_type == 'gb':
                model = GradientBoostingRegressor(random_state=42)
            elif model_type == 'ada':
                model = AdaBoostRegressor(random_state=42)
            elif model_type == 'nn':
                model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            metric_name = 'R² Score'
        
        # Fit and evaluate
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        if is_categorical:
            score = accuracy_score(y_test, y_pred)
            if class_names is not None:
                class_report = classification_report(y_test, y_pred, target_names=class_names)
            else:
                class_report = classification_report(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
            class_report = None
        
        mse = mean_squared_error(y_test, y_pred)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:  # For multi-class logistic regression
                importances = importances.mean(axis=0)
        else:
            importances = None
        
        # Create enhanced results display with AI/ML theme
        results = html.Div([
            dbc.Alert([
                html.H4([
                    html.I(className="fas fa-check-circle me-2"),
                    "Model Training Successful!"
                ], className="alert-heading mb-3"),
                html.Hr(),
                html.P([
                    html.I(className="fas fa-brain me-2"),
                    f"Model Architecture: {model.__class__.__name__}"
                ], className="mb-0")
            ], color="success", className="mb-4 fade-in"),
            
            dbc.Row([
                # Model Performance Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-line me-2"),
                            "Performance Metrics"
                        ], className="bg-primary text-white"),
                        dbc.CardBody([
                            html.Div([
                                html.H5(f"{metric_name}: {score:.4f}", className="mb-3"),
                                dbc.Progress(
                                    value=score * 100 if metric_name == 'Accuracy' else max(min(score * 100, 100), 0),
                                    color="success",
                                    striped=True,
                                    animated=True,
                                    className="mb-3"
                                ),
                                html.P([
                                    html.I(className="fas fa-square-root-alt me-2"),
                                    f"Mean Squared Error: {mse:.4f}"
                                ], className="mb-3"),
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-clipboard-list me-2"),
                                        "Classification Report:"
                                    ], className="mt-3 mb-2"),
                                    html.Pre(class_report, className="bg-light p-3 rounded")
                                ]) if class_report else None,
                            ])
                        ])
                    ], className="h-100 shadow-sm")
                ], width=6),
                
                # Feature Importance Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-weight-hanging me-2"),
                            "Feature Importance"
                        ], className="bg-info text-white"),
                        dbc.CardBody([
                            html.Div([
                                html.Div([
                                    html.Strong([
                                        html.I(className="fas fa-code-branch me-2"),
                                        f"{feature}: "
                                    ]),
                                    dbc.Progress(
                                        value=float(importance) * 100,
                                        color="success",
                                        striped=True,
                                        animated=True,
                                        label=f"{float(importance):.3f}",
                                        className="mb-2"
                                    )
                                ]) for feature, importance in zip(features, importances)
                            ]) if importances is not None else html.P("Feature importance not available for this model type", className="text-muted")
                        ])
                    ], className="h-100 shadow-sm")
                ], width=6)
            ], className="fade-in")
        ])
        
        return results
        
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.H4([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Error in Model Training"
                ], className="alert-heading mb-3"),
                html.Hr(),
                html.P(f"Details: {str(e)}", className="mb-3"),
                html.Hr(),
                html.Div([
                    html.P([
                        html.I(className="fas fa-lightbulb me-2"),
                        "Suggestions:"
                    ], className="mb-2"),
                    html.Ul([
                        html.Li("Check if your target variable is appropriate for the selected model type"),
                        html.Li("Ensure all features are numeric or properly encoded"),
                        html.Li("Remove or handle any missing values in your data"),
                        html.Li("For datetime features, consider using simpler numeric representations")
                    ], className="mb-0")
                ])
            ], color="danger", className="mb-0 fade-in")
        ], className="py-4")

@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('dist-feature-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_distribution_plot(selected_feature, stored_data):
    if not selected_feature or stored_data is None:
        return create_empty_figure("Select a feature for distribution analysis")
    
    df = pd.read_json(stored_data, orient='split')
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df[selected_feature],
        name='Histogram',
        nbinsx=30,
        marker_color='#18BC9C',
        opacity=0.7
    ))
    
    # Add KDE
    if df[selected_feature].dtype in ['int64', 'float64']:
        kde = gaussian_kde(df[selected_feature].dropna())
        x_range = np.linspace(df[selected_feature].min(), df[selected_feature].max(), 100)
        kde_vals = kde(x_range)
        
        hist_vals, _ = np.histogram(df[selected_feature].dropna(), bins=30)
        scaling_factor = hist_vals.max() / kde_vals.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_vals * scaling_factor,
            name='Density',
            line=dict(color='#E74C3C', width=2),
            mode='lines'
        ))
    
    # Add box plot
    fig.add_trace(go.Box(
        x=df[selected_feature],
        name='Box Plot',
        marker_color='#3498DB',
        boxpoints='outliers',
        boxmean=True
    ))
    
    fig.update_layout(
        title=f'Distribution Analysis of {selected_feature}',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return apply_common_layout(fig)

@app.callback(
    Output('relationship-plot', 'figure'),
    [Input('relationship-x-selector', 'value'),
     Input('relationship-y-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_relationship_plot(x_feature, y_feature, stored_data):
    if not x_feature or not y_feature or stored_data is None:
        return create_empty_figure("Select X and Y variables for relationship analysis")
    
    try:
        df = pd.read_json(stored_data, orient='split')
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=df[x_feature],
            y=df[y_feature],
            mode='markers',
            name='Data Points',
            marker=dict(
                size=8,
                color='#18BC9C',
                opacity=0.6,
                line=dict(
                    color='#2C3E50',
                    width=1
                )
            )
        ))
        
        # Add trend line
        z = np.polyfit(df[x_feature], df[y_feature], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[x_feature].min(), df[x_feature].max(), 100)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=p(x_range),
            name='Trend Line',
            line=dict(color='#E74C3C', width=2, dash='dash'),
            mode='lines'
        ))
        
        # Add moving average
        window_size = min(20, len(df) // 5)
        df_sorted = df.sort_values(by=x_feature)
        moving_avg = df_sorted[y_feature].rolling(window=window_size, center=True).mean()
        
        fig.add_trace(go.Scatter(
            x=df_sorted[x_feature],
            y=moving_avg,
            name=f'{window_size}-point Moving Average',
            line=dict(color='#3498DB', width=2),
            mode='lines'
        ))
        
        # Add marginal distributions
        # X-axis histogram
        fig.add_trace(go.Histogram(
            x=df[x_feature],
            name=f'{x_feature} Distribution',
            yaxis='y2',
            marker_color='#18BC9C',
            opacity=0.7,
            showlegend=False
        ))
        
        # Y-axis histogram
        fig.add_trace(go.Histogram(
            y=df[y_feature],
            name=f'{y_feature} Distribution',
            xaxis='x2',
            marker_color='#18BC9C',
            opacity=0.7,
            showlegend=False
        ))
        
        # Update layout with subplots
        fig.update_layout(
            title=f'Relationship between {x_feature} and {y_feature}',
            xaxis=dict(
                domain=[0, 0.85],
                showgrid=True,
                title=x_feature
            ),
            yaxis=dict(
                domain=[0, 0.85],
                showgrid=True,
                title=y_feature
            ),
            xaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100)
        )
        
        return apply_common_layout(fig)
        
    except Exception as e:
        print(f"Error in relationship plot: {str(e)}")
        return create_empty_figure(f"Error creating relationship plot: {str(e)}")

@app.callback(
    Output('correlation-plot', 'figure'),
    [Input('correlation-features-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_correlation_plot(selected_features, stored_data):
    if not selected_features or len(selected_features) < 2 or stored_data is None:
        return create_empty_figure("Select at least two features for correlation analysis")
    
    try:
        df = pd.read_json(stored_data, orient='split')
        
        # Calculate correlation matrix
        corr_matrix = df[selected_features].corr()
        
        # Create figure
        fig = go.Figure()
        
        # Add heatmap with improved styling
        fig.add_trace(go.Heatmap(
            z=corr_matrix,
            x=selected_features,
            y=selected_features,
            colorscale=[
                [0.0, '#C0392B'],  # Strong negative correlation
                [0.5, '#ECF0F1'],  # No correlation
                [1.0, '#18BC9C']   # Strong positive correlation
            ],
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 12, "color": "#2C3E50"},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text='Correlation',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12),
                len=0.9,
                thickness=15,
                bgcolor='rgba(255,255,255,0.9)',
                borderwidth=1,
                bordercolor='#2C3E50'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Correlation Analysis',
                x=0.5,
                font=dict(size=24, color='#2C3E50')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=max(500, len(selected_features) * 70),
            height=max(500, len(selected_features) * 70),
            margin=dict(t=100, l=100, r=40, b=60),
            showlegend=False,
            xaxis=dict(
                tickangle=45,
                tickfont=dict(size=12),
                gridcolor='#ECF0F1',
                showgrid=True,
                title=dict(text="Features", font=dict(size=14))
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                gridcolor='#ECF0F1',
                showgrid=True,
                title=dict(text="Features", font=dict(size=14))
            )
        )
        
        # Add correlation values as annotations
        annotations = []
        for i in range(len(selected_features)):
            for j in range(len(selected_features)):
                annotations.append(dict(
                    x=selected_features[i],
                    y=selected_features[j],
                    text=f"{corr_matrix.iloc[j, i]:.2f}",
                    showarrow=False,
                    font=dict(
                        size=12,
                        color='black' if abs(corr_matrix.iloc[j, i]) < 0.7 else 'white'
                    )
                ))
        
        fig.update_layout(annotations=annotations)
        return fig
        
    except Exception as e:
        print(f"Error in correlation plot: {str(e)}")
        return create_empty_figure(f"Error creating correlation plot: {str(e)}")

@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('time-feature-selector', 'value'),
     Input('value-feature-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_time_series_plot(time_feature, value_feature, stored_data):
    if not time_feature or not value_feature or stored_data is None:
        return create_empty_figure("Select time and value features for time series analysis")
    
    df = pd.read_json(stored_data, orient='split')
    
    fig = go.Figure()
    
    # Add main line plot
    fig.add_trace(go.Scatter(
        x=df[time_feature],
        y=df[value_feature],
        name='Actual',
        line=dict(color='#18BC9C', width=2)
    ))
    
    # Add rolling average
    window_size = max(5, len(df) // 20)
    rolling_avg = df[value_feature].rolling(window=window_size).mean()
    fig.add_trace(go.Scatter(
        x=df[time_feature],
        y=rolling_avg,
        name=f'{window_size}-point Moving Average',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Time Series Analysis: {value_feature} over {time_feature}'
    )
    
    return apply_common_layout(fig)

@app.callback(
    Output('comparative-plot', 'figure'),
    [Input('group-feature-selector', 'value'),
     Input('compare-feature-selector', 'value'),
     Input('stored-data', 'data')]
)
def update_comparative_plot(group_feature, compare_feature, stored_data):
    if not group_feature or not compare_feature or stored_data is None:
        return create_empty_figure("Select grouping and comparison features")
    
    df = pd.read_json(stored_data, orient='split')
    
    fig = go.Figure()
    
    # Add violin plots
    fig.add_trace(go.Violin(
        x=df[group_feature],
        y=df[compare_feature],
        name='Distribution',
        box_visible=True,
        meanline_visible=True,
        points='outliers'
    ))
    
    # Add box plots
    fig.add_trace(go.Box(
        x=df[group_feature],
        y=df[compare_feature],
        name='Box Plot',
        marker_color='#3498DB',
        boxpoints='outliers',
        boxmean=True
    ))
    
    fig.update_layout(
        title=f'Comparative Analysis: {compare_feature} by {group_feature}'
    )
    
    return apply_common_layout(fig)

def create_empty_figure(message):
    return {
        'data': [],
        'layout': {
            'title': {
                'text': message,
                'font': {'size': 24, 'color': '#2C3E50'}
            },
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'annotations': [{
                'text': '📊 Select variables to visualize',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20, 'color': '#7f8c8d'}
            }]
        }
    }

def apply_common_layout(fig):
    colors = {
        'primary': '#18BC9C',
        'secondary': '#2C3E50',
        'accent': '#E74C3C',
        'light': '#ECF0F1'
    }
    
    # Create new layout settings
    new_layout = {
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'font': {
            'family': 'Arial',
            'size': 12,
            'color': colors['secondary']
        },
        'margin': dict(l=60, r=40, t=80, b=60),
        'showlegend': True,
        'legend': dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=colors['secondary'],
            borderwidth=1
        ),
        'xaxis': dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['secondary'],
            tickfont=dict(size=12)
        ),
        'yaxis': dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#ECF0F1',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor=colors['secondary'],
            tickfont=dict(size=12)
        )
    }
    
    # Update layout while preserving any existing settings
    if hasattr(fig, 'layout') and fig.layout is not None:
        current_layout = fig.layout
        for key, value in new_layout.items():
            if key not in current_layout:
                current_layout[key] = value
    else:
        fig.update_layout(new_layout)
    
    return fig

# Update the feature selectors when data is uploaded
@app.callback(
    [Output('dist-feature-selector', 'options'),
     Output('relationship-x-selector', 'options'),
     Output('relationship-y-selector', 'options'),
     Output('correlation-features-selector', 'options'),
     Output('time-feature-selector', 'options'),
     Output('value-feature-selector', 'options'),
     Output('group-feature-selector', 'options'),
     Output('compare-feature-selector', 'options')],
    [Input('stored-data', 'data')]
)
def update_feature_selectors(stored_data):
    if stored_data is None:
        empty_options = []
        return [empty_options] * 8
    
    df = pd.read_json(stored_data, orient='split')
    options = [{'label': col, 'value': col} for col in df.columns]
    return [options] * 8

# Update the model comparison callback
@app.callback(
    Output('model-comparison-results', 'children'),
    [Input('model-comparison-selector', 'value'),
     Input('stored-data', 'data'),
     Input('target-selector', 'value'),
     Input('feature-selector', 'value')]
)
def compare_models(selected_models, stored_data, target, features):
    if not selected_models or not stored_data or not target or not features:
        return html.Div([
            html.I(className="fas fa-balance-scale fa-3x text-primary mb-3"),
            html.H4("Model Comparison Ready", className="mb-3"),
            html.P([
                "Select models from the dropdown above to compare their performance",
                html.Br(),
                "You can select multiple models to compare"
            ], className="text-muted")
        ], className="text-center py-4 fade-in")
    
    try:
        # Load and prepare data
        df = pd.read_json(stored_data, orient='split')
        X = df[features].copy()
        y = df[target].copy()
        
        # Function to convert datetime to numeric
        def convert_datetime(series):
            if pd.api.types.is_datetime64_any_dtype(series):
                return pd.to_numeric(series.astype(np.int64))
            return series

        # Handle different data types in features
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = convert_datetime(X[col])
            elif pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object':
                X[col] = X[col].fillna(X[col].mode()[0])
                X[col] = pd.Categorical(X[col]).codes
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].mean())
        
        # Handle target variable
        if pd.api.types.is_datetime64_any_dtype(y):
            y = convert_datetime(y)
            is_categorical = False
            class_names = None
        else:
            # Check if target is categorical
            unique_values = pd.unique(y)
            is_categorical = (y.dtype == 'object' or 
                            pd.api.types.is_categorical_dtype(y) or 
                            (len(unique_values) < 10 and not np.issubdtype(y.dtype, np.floating)))
            
            if is_categorical:
                # Handle missing values in target
                y = y.fillna(y.mode()[0])
                # Store original class names before encoding
                class_names = np.array([str(val) for val in unique_values])
                # Encode categorical target
                le = LabelEncoder()
                y = le.fit_transform(y)
            else:
                # Convert to numeric and handle missing values
                y = pd.to_numeric(y, errors='coerce')
                y = y.fillna(y.mean())
                class_names = None
        
        # Ensure all data is numeric
        X = X.astype(float)
        y = y.astype(float)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Initialize results storage
        results = []
        confusion_matrices = []
        roc_curves = []
        
        # Model mapping dictionary
        model_mapping = {
            'classification': {
                'linear': lambda: LogisticRegression(max_iter=1000, random_state=42),
                'rf': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': lambda: SVC(kernel='rbf', probability=True, random_state=42),
                'knn': lambda: KNeighborsClassifier(n_neighbors=5),
                'dt': lambda: DecisionTreeClassifier(random_state=42),
                'gb': lambda: GradientBoostingClassifier(random_state=42),
                'ada': lambda: AdaBoostClassifier(random_state=42),
                'nb': lambda: GaussianNB(),
                'nn': lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            },
            'regression': {
                'linear': lambda: LinearRegression(),
                'rf': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
                'svm': lambda: SVR(kernel='rbf'),
                'knn': lambda: KNeighborsRegressor(n_neighbors=5),
                'dt': lambda: DecisionTreeRegressor(random_state=42),
                'gb': lambda: GradientBoostingRegressor(random_state=42),
                'ada': lambda: AdaBoostRegressor(random_state=42),
                'nn': lambda: MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            }
        }
        
        # Train and evaluate each selected model
        for model_type in selected_models:
            # Initialize the appropriate model
            if is_categorical:
                if model_type in model_mapping['classification']:
                    model = model_mapping['classification'][model_type]()
                else:
                    continue
            else:
                if model_type in model_mapping['regression']:
                    model = model_mapping['regression'][model_type]()
                else:
                    continue
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()
            
            # Fit model for additional metrics
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            if is_categorical:
                # Calculate confusion matrix
                cm = confusion_matrix(y, y_pred)
                confusion_matrices.append({
                    'model': model_type,
                    'matrix': cm
                })
                
                # Calculate ROC curve if binary classification
                if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_scaled)[:, 1]
                    fpr, tpr, _ = roc_curve(y, y_prob)
                    roc_auc = auc(fpr, tpr)
                    roc_curves.append({
                        'model': model_type,
                        'fpr': fpr,
                        'tpr': tpr,
                        'auc': roc_auc
                    })
            
            # Store results
            results.append({
                'model': model_type,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'model_obj': model
            })
        
        if not results:
            return html.Div([
                dbc.Alert([
                    html.H4([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "No Valid Models"
                    ], className="alert-heading mb-3"),
                    html.P("None of the selected models are compatible with your data type.")
                ], color="warning", className="mb-0 fade-in")
            ], className="py-4")
        
        # Create comparison visualization
        comparison_fig = go.Figure()
        for result in results:
            comparison_fig.add_trace(go.Bar(
                x=[result['model']],
                y=[result['mean_cv_score']],
                error_y=dict(
                    type='data',
                    array=[result['std_cv_score']],
                    visible=True
                ),
                name=result['model'],
                text=[f"{result['mean_cv_score']:.3f} ± {result['std_cv_score']:.3f}"],
                textposition='auto',
            ))
        
        comparison_fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Cross-Validation Score',
            barmode='group',
            showlegend=False
        )
        
        # Create confusion matrices visualization if classification
        confusion_fig = None
        if is_categorical and confusion_matrices:
            confusion_fig = go.Figure()
            for cm_data in confusion_matrices:
                confusion_fig.add_trace(go.Heatmap(
                    z=cm_data['matrix'],
                    x=class_names if class_names is not None else np.unique(y),
                    y=class_names if class_names is not None else np.unique(y),
                    text=cm_data['matrix'],
                    texttemplate='%{text}',
                    name=cm_data['model'],
                    visible=False
                ))
            
            # Make first confusion matrix visible
            if confusion_fig.data:
                confusion_fig.data[0].visible = True
            
            confusion_fig.update_layout(
                title='Confusion Matrices',
                updatemenus=[{
                    'buttons': [
                        dict(
                            args=[{"visible": [i == j for j in range(len(confusion_matrices))]}],
                            label=cm_data['model'],
                            method="update"
                        ) for i, cm_data in enumerate(confusion_matrices)
                    ],
                    'direction': 'right',
                    'showactive': True,
                    'x': 0.5,
                    'y': 1.15,
                    'xanchor': 'center'
                }]
            )
        
        # Create ROC curves visualization if binary classification
        roc_fig = None
        if roc_curves:
            roc_fig = go.Figure()
            for roc_data in roc_curves:
                roc_fig.add_trace(go.Scatter(
                    x=roc_data['fpr'],
                    y=roc_data['tpr'],
                    name=f"{roc_data['model']} (AUC = {roc_data['auc']:.3f})",
                    mode='lines'
                ))
            
            roc_fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name='Random Chance',
                line=dict(dash='dash')
            ))
            
            roc_fig.update_layout(
                title='ROC Curves',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
        
        # Create feature importance comparison
        importance_fig = None
        if any(hasattr(result['model_obj'], 'feature_importances_') for result in results):
            importance_fig = go.Figure()
            for result in results:
                if hasattr(result['model_obj'], 'feature_importances_'):
                    importances = result['model_obj'].feature_importances_
                    importance_fig.add_trace(go.Bar(
                        x=features,
                        y=importances,
                        name=result['model']
                    ))
            
            importance_fig.update_layout(
                title='Feature Importance Comparison',
                xaxis_title='Features',
                yaxis_title='Importance',
                barmode='group'
            )
        
        # Return comparison results
        return html.Div([
            # Performance Comparison
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Performance Comparison"
                        ], className="bg-primary text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=comparison_fig,
                                config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                className="border rounded shadow-sm"
                            )
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            # Confusion Matrices
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-th me-2"),
                            "Confusion Matrices"
                        ], className="bg-info text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=confusion_fig,
                                config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                className="border rounded shadow-sm"
                            ) if confusion_fig else html.P("Confusion matrices not available for regression tasks", className="text-muted")
                        ])
                    ], className="mb-4")
                ], width=12)
            ]) if is_categorical else None,
            
            # ROC Curves
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-line me-2"),
                            "ROC Curves"
                        ], className="bg-success text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=roc_fig,
                                config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                className="border rounded shadow-sm"
                            ) if roc_fig else html.P("ROC curves only available for binary classification", className="text-muted")
                        ])
                    ], className="mb-4")
                ], width=12)
            ]) if roc_curves else None,
            
            # Feature Importance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-weight-hanging me-2"),
                            "Feature Importance Comparison"
                        ], className="bg-warning text-white"),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=importance_fig,
                                config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False},
                                className="border rounded shadow-sm"
                            ) if importance_fig else html.P("Feature importance not available for all selected models", className="text-muted")
                        ])
                    ])
                ], width=12)
            ]) if importance_fig else None
        ])
        
    except Exception as e:
        return html.Div([
            dbc.Alert([
                html.H4([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Error in Model Comparison"
                ], className="alert-heading mb-3"),
                html.Hr(),
                html.P(f"Details: {str(e)}", className="mb-3"),
                html.Hr(),
                html.Div([
                    html.P([
                        html.I(className="fas fa-lightbulb me-2"),
                        "Suggestions:"
                    ], className="mb-2"),
                    html.Ul([
                        html.Li("Ensure your data is properly formatted"),
                        html.Li("Check if all selected models are appropriate for your data type"),
                        html.Li("Verify that your target variable is correctly specified"),
                        html.Li("Make sure all features are properly encoded")
                    ], className="mb-0")
                ])
            ], color="danger", className="mb-0 fade-in")
        ], className="py-4")

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 