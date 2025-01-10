from dash import Dash, html, dcc, Input, Output, callback_context
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# Load the pre-processed dataset
df = pd.read_csv('df4.csv')

# Feature groups
preferences = df[['vendor_loyalty_score', 'relative_cuisine_variety', 'chain_consumption']]
behaviours = df[['first_order', 'days_since_last_order', 'order_frequency', 'total_orders',
                 'total_amount_spent', 'average_spending']]

# Columns available for visualization
df_columns = [col for col in df.columns if col not in ['merged_labels']]

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'],
           suppress_callback_exceptions=True)

# Main page layout
main_page_layout = html.Div(
    style={
        'height': '100vh',
        'background': '#008080',
        'color': '#ffffff',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'textAlign': 'center',
        'fontFamily': 'Roboto, sans-serif'
    },
    children=[
        html.H1('ABCDEats Inc. Customer Segmentation', style={'fontSize': '50px', 'fontWeight': '700'}),
        html.Div(
            style={'marginTop': '50px', 'display': 'flex', 'justifyContent': 'center', 'gap': '20px'},
            children=[
                dcc.Link(
                    html.Button('Cluster Exploration 🌌', className='btn btn-lg btn-light'),
                    href='/tsne'
                ),
                dcc.Link(
                    html.Button('Visualization Tools 📊', className='btn btn-lg btn-dark'),
                    href='/compare'
                )
            ]
        ),
    ]
)

# Cluster Exploration layout
cluster_exploration_page_layout = html.Div(style={'padding': '20px'}, children=[
    html.H1('Cluster Exploration', className='text-center mb-4'),
    dcc.Dropdown(
        id='cluster-select',
        options=[{'label': f'Cluster {i}', 'value': i} for i in df['merged_labels'].unique()],
        value=df['merged_labels'].unique()[0],
        placeholder="Select a Cluster"
    ),
    dcc.Dropdown(
        id='feature-select',
        options=[
            {'label': 'Preferences-Based', 'value': 'preferences'},
            {'label': 'Behavioral-Based', 'value': 'behavioral'}
        ],
        value='preferences',
        placeholder="Select Feature Group"
    ),
    dcc.Graph(id='cluster-graph'),
    html.Div(id='cluster-summary', style={'marginTop': '40px'}),
    dcc.Link(html.Button('Back to Main Page', className='btn btn-warning mt-4'), href='/')
])

@app.callback(
    [Output('cluster-graph', 'figure'),
     Output('cluster-summary', 'children')],
    [Input('feature-select', 'value'),
     Input('cluster-select', 'value')]
)
def update_cluster_visuals(feature_group, selected_cluster):
    features = preferences.columns if feature_group == 'preferences' else behaviours.columns
    cluster_data = df[df['merged_labels'] == selected_cluster]
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(cluster_data[features])
    cluster_data['TSNE1'], cluster_data['TSNE2'] = tsne_results[:, 0], tsne_results[:, 1]

    fig = px.scatter(cluster_data, x='TSNE1', y='TSNE2', color='merged_labels',
                     title=f"t-SNE Plot for Cluster {selected_cluster}")

    cluster_summary = html.Table([
        html.Tr([html.Th('Cluster'), html.Td(selected_cluster)]),
        html.Tr([html.Th('Number of Data Points'), html.Td(len(cluster_data))]),
    ])

    return fig, cluster_summary

# Visualization Tools layout
compare_page_layout = html.Div(style={'padding': '20px'}, children=[
    html.H1('Visualization Tools 📊', className='text-center mb-4'),
    dcc.Dropdown(
        id='chart-feature-select',
        options=[{'label': col, 'value': col} for col in df_columns],
        placeholder="Select Feature for Visualization",
        style={'width': '50%'}
    ),
    dcc.Dropdown(
        id='heatmap-feature-select',
        options=[{'label': col, 'value': col} for col in df_columns],
        multi=True,
        placeholder="Select Features for Heatmap",
        style={'width': '50%', 'marginTop': '20px'}
    ),
    html.Div(
        children=[
            html.Button('Box Plot', id='box-plot-button', n_clicks=0, className='btn btn-secondary'),
            html.Button('Histogram', id='histogram-button', n_clicks=0, className='btn btn-info'),
            html.Button('Line Chart', id='line-chart-button', n_clicks=0, className='btn btn-warning'),
            html.Button('Heatmap', id='heatmap-button', n_clicks=0, className='btn btn-danger')
        ],
        style={'marginTop': '20px', 'display': 'flex', 'justifyContent': 'center', 'gap': '10px'}
    ),
    dcc.Graph(id='visualization-output'),
    dcc.Link(html.Button('Back to Main Page', className='btn btn-warning mt-4'), href='/')
])

@app.callback(
    Output('visualization-output', 'figure'),
    [Input('chart-feature-select', 'value'),
     Input('heatmap-feature-select', 'value'),
     Input('box-plot-button', 'n_clicks'),
     Input('histogram-button', 'n_clicks'),
     Input('line-chart-button', 'n_clicks'),
     Input('heatmap-button', 'n_clicks')]
)
def update_visualization(selected_feature, heatmap_features, box_clicks, hist_clicks, line_clicks, heatmap_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return px.scatter()

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'heatmap-button' and heatmap_features:
        filtered_data = df[heatmap_features].select_dtypes(include=['number'])
        corr_matrix = filtered_data.corr()
        return px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', title="Feature Correlation Heatmap")

    if button_id == 'box-plot-button' and selected_feature:
        return px.box(df, y=selected_feature, color='merged_labels', title=f'{selected_feature} Box Plot')

    if button_id == 'histogram-button' and selected_feature:
        return px.histogram(df, x=selected_feature, nbins=30, title=f'{selected_feature} Histogram')

    if button_id == 'line-chart-button' and selected_feature and 'first_order' in df.columns:
        return px.line(df, x='first_order', y=selected_feature, title=f'{selected_feature} Over Time')

    return px.scatter()

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/tsne':
        return cluster_exploration_page_layout
    elif pathname == '/compare':
        return compare_page_layout
    return main_page_layout

if __name__ == '__main__':
    app.run_server(debug=True)
