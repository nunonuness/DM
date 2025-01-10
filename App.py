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

df_columns = ['customer_id', 'CUI_American', 'CUI_Asian', 'CUI_Beverages', 'CUI_Cafe',
       'CUI_Chicken Dishes', 'CUI_Chinese', 'CUI_Desserts', 'CUI_Healthy',
       'CUI_Indian', 'CUI_Italian', 'CUI_Japanese', 'CUI_Noodle Dishes',
       'CUI_OTHER', 'CUI_Street Food / Snacks', 'CUI_Thai', 'DOW_0', 'DOW_1',
       'DOW_2', 'DOW_3', 'DOW_4', 'DOW_5', 'DOW_6', 'HR_0', 'HR_1', 'HR_10',
       'HR_11', 'HR_12', 'HR_13', 'HR_14', 'HR_15', 'HR_16', 'HR_17', 'HR_18',
       'HR_19', 'HR_2', 'HR_20', 'HR_21', 'HR_22', 'HR_23', 'HR_3', 'HR_4',
       'HR_5', 'HR_6', 'HR_7', 'HR_8', 'HR_9', 'average_spending',
       'customer_age', 'days_since_last_order', 'first_order', 'is_chain',
       'last_order', 'order_frequency', 'product_count', 'total_amount_spent',
       'total_orders', 'vendor_count', 'vendor_loyalty_score',
       'relative_cuisine_variety', 'chain_consumption', 'customer_city',
       'available_cuisines_city', 'cuisine_variety', 'last_promo',
       'payment_method', 'age_group', 'customer_lifecycle_stage',
       'peak_order_day', 'peak_order_hour']

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
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px',
                                       'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/tsne'
                ),
                # Button 2
                dcc.Link(
                    html.Button('Visualization Tools 📊',
                                className='btn btn-lg btn-dark',
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px',
                                       'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/compare'
                ),
                # Button 3
                dcc.Link(
                    html.Button('Filter Data 🔍',
                                className='btn btn-lg btn-warning',
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px',
                                       'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
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

    # Pie chart for additional visual
    dcc.Graph(id='cluster-distribution-pie'),

    # Cluster summary table
    html.Div(id='cluster-summary', style={'marginTop': '40px'}),

    # Back button
    dcc.Link(html.Button('Back to Main Page', className='btn btn-warning mt-4'), href='/')
])

# Define the layout for the Compare page
compare_page_layout = html.Div(style={'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
    html.H1('Visualization', className='text-center mb-4'),

    # Dropdown for feature selection
    dcc.Dropdown(
        id='heatmap-feature-select',
        options=[{'label': col, 'value': col} for col in df.columns if col not in ['merged_labels']],
        multi=True,
        placeholder="Select Features for Heatmap"
    ),

    # Heatmap display
    dcc.Graph(id='heatmap-graph'),

    # Back button
    dcc.Link(html.Button('Back to Main Page', className='btn btn-warning mt-4'), href='/')
])

# Callbacks for the Cluster Exploration page
@app.callback(
    [Output('cluster-graph', 'figure'),
     Output('cluster-summary', 'children'),
     Output('cluster-distribution-pie', 'figure')],
    [Input('feature-select', 'value'),
     Input('cluster-select', 'value')]
)
def update_cluster_visuals(feature_group, selected_cluster):
    try:
        # Filter data based on feature group selection
        features = preferences.columns if feature_group == 'preferences' else behaviours.columns

        # Filter data by cluster selection
        cluster_data = df[df['merged_labels'] == selected_cluster]

        # Apply t-SNE on the selected features
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(cluster_data[features])

        # Add the t-SNE results to the dataframe
        cluster_data['TSNE1'] = tsne_results[:, 0]
        cluster_data['TSNE2'] = tsne_results[:, 1]

        # Create the interactive t-SNE plot
        fig = px.scatter(cluster_data, x='TSNE1', y='TSNE2', color='merged_labels',
                         title=f"t-SNE Plot for Cluster {selected_cluster}",
                         labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                         color_continuous_scale='Viridis')

        # Create a summary of the cluster
        cluster_summary = html.Table([
            html.Tr([html.Th('Cluster'), html.Td(selected_cluster)]),
            html.Tr([html.Th('Number of Data Points'), html.Td(len(cluster_data))]),
        ])

        # Create a pie chart
        pie_fig = px.pie(cluster_data, names='relative_cuisine_variety',
                         title=f"Cluster {selected_cluster} Cuisine Variety Distribution")

        return fig, cluster_summary, pie_fig
    except Exception as e:
        return {}, "Error generating visuals. Check feature selection.", {}

# Callbacks for the Compare page (Heatmap)
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('heatmap-feature-select', 'value')]
)
def update_heatmap(selected_features):
    # Check if the selected features are valid
    if not selected_features:
        return px.imshow([], title="No Features Selected for Heatmap", labels={'color': 'Correlation'})

    try:
        # Filter the DataFrame to include only selected features
        filtered_data = df[selected_features]
        
        # Ensure all selected columns are numeric
        numeric_data = filtered_data.select_dtypes(include=['number'])

        if numeric_data.empty:
            return px.imshow([], title="No Numeric Features Found", labels={'color': 'Correlation'})

        # Compute the correlation matrix
        corr_matrix = numeric_data.corr()

        # Generate the heatmap figure
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis',
                        title="Feature Correlation Heatmap",
                        labels={'color': 'Correlation'})

        return fig

    except Exception as e:
        # Log the exception and return a blank figure with error message
        print(f"Error in heatmap generation: {e}")
        return px.imshow([], title=f"Error Generating Heatmap: {str(e)}", labels={'color': 'Correlation'})




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
        return compare_page_layout
    else:
        return main_page_layout

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
