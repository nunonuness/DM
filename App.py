from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# pre-processed dataset (contains the required features and cluster labels)
df = pd.read_csv('df4.csv')

# based characteristics features
preferences = df[['vendor_loyalty_score', 'relative_cuisine_variety', 'chain_consumption']]
behaviours = df[['first_order', 'days_since_last_order', 'order_frequency', 'total_orders',
                 'total_amount_spent', 'average_spending']]

# features without cluster labels
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

# initialize the app with Bootstrap CSS and suppress callback exceptions
app = Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'],
           suppress_callback_exceptions=True)

# layout for the main page
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
        'fontFamily': 'Roboto, sans-serif',
        'overflow': 'hidden'
    },
    children=[
        html.H1('ABCDEats Inc. Customer Segmentation',
                style={'fontSize': '50px', 'fontWeight': '700', 'letterSpacing': '2px'}),
        html.Div(
            style={'marginTop': '50px', 'display': 'flex', 'justifyContent': 'center', 'gap': '20px'},
            children=[
                dcc.Link(
                    html.Button('Cluster Exploration 🌌',
                                className='btn btn-lg btn-light',
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px',
                                       'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/tsne'
                ),
                dcc.Link(
                    html.Button('Visualization Tools 📊',
                                className='btn btn-lg btn-dark',
                                style={'fontSize': '20px', 'padding': '20px 60px', 'borderRadius': '30px',
                                       'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.15)', 'transition': 'all 0.3s ease'}),
                    href='/compare'
                )
            ]
        ),
    ]
)

# layout for the Cluster Exploration page
cluster_exploration_page_layout = html.Div(style={'padding': '20px', 'maxWidth': '1200px', 'margin': 'auto'}, children=[
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
    try:
        features = preferences.columns if feature_group == 'preferences' else behaviours.columns
        cluster_data = df[df['merged_labels'] == selected_cluster]
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(cluster_data[features])
        cluster_data['TSNE1'], cluster_data['TSNE2'] = tsne_results[:, 0], tsne_results[:, 1]

        fig = px.scatter(cluster_data, x='TSNE1', y='TSNE2', color='merged_labels',
                         title=f"t-SNE Plot for Cluster {selected_cluster}",
                         labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                         color_continuous_scale='Viridis')

        cluster_summary = html.Table([
            html.Tr([html.Th('Cluster'), html.Td(selected_cluster)]),
            html.Tr([html.Th('Number of Data Points'), html.Td(len(cluster_data))]),
        ])

        return fig, cluster_summary
    except Exception as e:
        return {}, html.Div(f"Error: {str(e)}")

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/tsne':
        return cluster_exploration_page_layout
    return main_page_layout

if __name__ == '__main__':
    app.run_server(debug=True)
