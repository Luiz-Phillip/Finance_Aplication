import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from stock import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {
    'background': '#143A5E',
    'text': '#FAF7F6'
        }
app.layout = html.Div(style={'backgroundColor': colors['background']}, children = [
    html.H1(children='Welcome to Finance Aplication',
            style = {
                'textAling' : 'center',
                'color' : colors['text']
            }),
    dcc.Input(
        id='input_stock',
        type='text',
        value='TOTS3',
        style={
            'backgroundColor': colors['background'],
            'color' : colors['text']}
    ),
    html.Button(id='submit_stock', n_clicks=0, children='Double Click for Submit', autoFocus = True)
    ,
     dcc.Tabs(id="tabs-example", value='tab-1-example', children=[
        dcc.Tab(label='Historical Data', value='tab_1'),
        dcc.Tab(label='Beta Indices', value='tab_2'),
    ],
        colors = { 'border': '#5DA8A1', 'primary': '#A85D65', 'background': '#D8DDE6' }),
    html.Div(id = 'stock')
    ])


@app.callback(
    Output('stock', 'children'), 
    [Input('submit_stock', 'n_clicks'), Input('input_stock', 'value'), Input('tabs-example', 'value')])
def callback_a(clicks, x, tabs):
    if 'atual_click' not in locals():
        atual_click = 0
    if (clicks > atual_click):
        stock = x+'.SA'
        graph = paper_value(paper = stock)
        atual_click = clicks
    if tabs == 'tab_1':
        return html.Div([dcc.Graph(figure = graph.visualization_action())])
    elif tabs == 'tab_2':
        return html.Div([dcc.Graph(figure = graph.visualization_cov_var())])

atual_click = 0
if __name__ == '__main__':
    
    app.run_server(debug=True)