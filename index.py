import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

# Connect to main app.py file
import dash
import dash_bootstrap_components as dbc

FA = "https://use.fontawesome.com/releases/v5.15.2/css/all.css"
app = dash.Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP,FA])
server = app.server
# from app import server

# Connect to your app pages
from apps import camera, recoginze_image
# server = app.server
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div(
            dbc.Button(dcc.Link('Recoginze_Camera', href='/apps/camera'),outline=True, color="success", className="me-1"),
            style={'height':'50px','width':'50%','paddingLeft':500}
        ),
        html.Div(
            dbc.Button(dcc.Link('Recoginze_Image', href='/apps/recoginze_image'), outline=True, color="warning", className="me-1"),
            style={'height':'50px','width':'50%'}
        ),
        # dbc.Button(dcc.Link('Recoginze_Image', href='/apps/recoginze_image'), color="dark", className="me-1",style={'height':'50px','width':'50%'}),
    ], style={'display':'flex','flex-direction':'row','height':50,'width':'100%'},className="row"),
    html.Div(id='page-content', children=[])
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/recoginze_image':
        return recoginze_image.layout
    if pathname == '/apps/camera':
        return camera.page_2_layout

