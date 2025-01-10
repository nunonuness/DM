from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# Load the pre-processed dataset (already contains the required features and cluster labels)
df = pd.read_csv('df4.csv')  # Replace with your actual file path

# Define the feature sets
preferences = df[['vendor_loyalty_score', 'relative_cuisine_variety', 'chain_consumption']]
behaviours = df[['first_order', 'days_since_last_order', 'order_frequency', 'total_orders', 
                 'total_amount_spent', 'average_spending']]

# Initialize the app with Bootstrap CSS and suppress callback exceptions
app = Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'],
           suppress_callback_exceptions=True)

# Define the layout for the Main page
main_page_layout = html.Div(
    style={
        'height': '100vh',  # Full viewport height
        'background': '#008080',
        'color': '#ffffff',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'textAlign': 'center',
        'fontFamily': 'Roboto, sans-serif',
        'overflow': 'hidden'
    },
    children=[
        html.H1('ABCDEats Inc. Customer Segmentation',  
                style={'fontSize': '50px', 'fontWeight': '700', 'letterSpacing': '2px'}),
        
        html.Div(
            style={'marginTop': '50px', 'display': 'flex', 'justifyContent': 'center', 'gap': '20px'},
            children=[
                # Button 1
                dcc.Link(
                    html.Button('Cluster Exploration 🌌', 
                                className='btn btn-lg btn-light', 
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/tsne'
                ),
                # Button 2
                dcc.Link(
                    html.Button('Compare Data 📊', 
                                className='btn btn-lg btn-dark', 
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/compare'
                ),
                # Button 3
                dcc.Link(
                    html.Button('Filter Data 🔍', 
                                className='btn btn-lg btn-warning', 
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/filter'
                )
            ]
        ),
    ]
)

# Define the layout for the Cluster Exploration page
cluster_exploration_page_layout = html.Div(style={'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
    html.H1('Cluster Exploration', className='text-center mb-4'),
    
    # Dropdown to select Cluster
    dcc.Dropdown(
        id='cluster-select',
        options=[{'label': f'Cluster {i}', 'value': i} for i in df['merged_labels'].unique()],
        value=df['merged_labels'].unique()[0],  # Default value
        placeholder="Select a Cluster"
    ),

    # Dropdown for feature group selection
    dcc.Dropdown(
        id='feature-select',
        options=[
            {'label': 'Preferences-Based', 'value': 'preferences'},
            {'label': 'Behavioral-Based', 'value': 'behavioral'}
        ],
        value='preferences',  # Default value
        placeholder="Select Feature Group"
    ),

    # Scatter plot to display clusters
    dcc.Graph(id='cluster-graph'),
    
    # Cluster summary table
    html.Div(id='cluster-summary', style={'marginTop': '40px'}),
    
    # Back button
    dcc.Link(html.Button('Back to Main Page', className='btn btn-warning mt-4'), href='/')
])

# Callback to update the cluster graph and summary based on the cluster and feature selection
@app.callback(
    [Output('cluster-graph', 'figure'),
     Output('cluster-summary', 'children')],
    [Input('feature-select', 'value'),
     Input('cluster-select', 'value')]
)
def update_tsne_plot(feature_group, selected_cluster):
    # Filter data based on feature group selection
    if feature_group == 'preferences':
        features = preferences.columns  # Use the actual column names from the preferences dataset
    else:
        features = behaviours.columns  # Use the actual column names from the behaviours dataset

    # Filter data by cluster selection
    cluster_data = df[df['merged_labels'] == selected_cluster]
    
    # Apply t-SNE on the selected features
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(cluster_data[features])
    
    # Add the t-SNE results to the dataframe
    cluster_data['TSNE1'] = tsne_results[:, 0]
    cluster_data['TSNE2'] = tsne_results[:, 1]
    
    # Create the interactive t-SNE plot using Plotly
    fig = px.scatter(cluster_data, x='TSNE1', y='TSNE2', color='merged_labels',
                     title=f"t-SNE Plot for Cluster {selected_cluster}",
                     labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                     color_continuous_scale='Viridis',
                     hover_data={  # Add hover data to make it more informative
                         'vendor_loyalty_score': True, 
                         'order_frequency': True, 
                         'relative_cuisine_variety': True,
                         'first_order': True,
                         'total_orders': True,
                         'total_amount_spent': True
                     })
    
    # Create a summary of the cluster
    cluster_summary = html.Table([
        html.Tr([html.Th('Cluster'), html.Td(selected_cluster)]),
        html.Tr([html.Th('Number of Data Points'), html.Td(len(cluster_data))]),
    ])

    return fig, cluster_summary

# Define the app layout with a location component for URL routing
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define the callback to update the page content based on the URL
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/tsne':
        return cluster_exploration_page_layout
    elif pathname == '/compare':
        return compare_page_layout  # Add compare page layout here
    elif pathname == '/filter':
        return filter_page_layout  # Add filter page layout here
    else:
        return main_page_layout

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
