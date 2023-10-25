import base64
import io

import pandas
import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
from flask_caching import Cache
from plotly.graph_objs import Figure
from plotly_resampler import register_plotly_resampler, FigureResampler

import predict_torque
from analyze_session import analyze_temperatures
from data import C, process_trace
import diskcache
import dash
from dash import DiskcacheManager, CeleryManager, Input, Output, html

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

CACHE_CONFIG = {
    # try 'FileSystemCache' if you don't want to setup redis
    'CACHE_TYPE': 'SimpleCache',
}
# cache = Cache()

config = {
    'toImageButtonOptions': {'height': None, 'width': None},
    'modeBarButtonsToAdd': ['drawline', 'eraseshape'],
    'doubleClick': 'reset'
}


def make_fig(test_data, filename):
    print("make_fig")
    fig = analyze_temperatures(test_data, filename)
    fig.update_layout(hoverlabel_namelength=-1)

    print("done_make_fig")
    return fig


app = Dash(__name__)
app.layout = html.Div([
    html.Div(
        id='filename',
        children=str("pick a file")),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        }),
    html.Hr(),
    dcc.Slider(
        id='lap-slider',
        min=0, max=900, step=1, marks=None),
    dcc.Graph(id='graph', config=config),
    dcc.Store(id='trace-hash'),
    dcc.Loading(dcc.Store(id='trace-json')),
    dcc.Loading(dcc.Store(id='figure-base')),
])


@dash.callback(
    Output('trace-json', 'data'),
    Output('trace-hash', 'data'),
    Output('filename', 'children'),
    Output('lap-slider', 'min'),
    Output('lap-slider', 'max'),
    Output('lap-slider', 'marks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    background=True,
    manager=background_callback_manager,

)
def update_trace_data(contents, filename):
    if filename is None:
        return None, None, ["Pick a file"], 0, 1, None
    print(f"update trace: {filename}")

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = process_trace(
                pandas.read_csv(
                    io.StringIO(
                        decoded.decode('utf-8')))
            )
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)

    df = df.sample(frac=.1)
    predict_torque.add_torque_prediction_trace(df)
    df = df.sample(frac=.1)
    df.sort_values(by=C.TIME_MERGED, inplace=True)

    json_df = df.to_json(orient='split')
    key = filename
    print(f"put key: {key}")
    # cache.cache.add(key, json_df)

    return json_df, key, [filename], min(df[C.TIME_MERGED]), max(df[C.TIME_MERGED]), None


@app.callback(
    Output('figure-base', 'data'),
    Input('trace-hash', 'data'),
    Input('trace-json', 'data'),
    State('upload-data', 'filename')
)
def update_figure(trace_hash, trace_json, filename):
    print(f"update_figure, {trace_hash}, {filename}")
    if trace_hash is None:
        return Figure()

    trace_frame = pandas.read_json(trace_json, orient='split')
    fig = make_fig(trace_frame, filename)

    return fig


# I don't know why this seems so fast all of a sudden.
app.clientside_callback(
    """
    function(figure, drag_value) {
        console.log("clientside drag" + drag_value)
                
        var layout = {
            ...figure.layout,
          shapes: [
            {
              type: 'vline',
              x0: drag_value,
              y0: 0,
              x1: drag_value,
              y1: 200,
              line: {
                color: 'rgb(55, 128, 191)',
                width: 3
              }
            },
            {
              type: 'line',
              x0: drag_value + 120,
              y0: 0,
              x1: drag_value + 120,
              y1: 200,
              line: {
                color: 'rgb(55, 128, 191)',
                width: 3
              }
            },
            {
              type: 'vline',
              x0: drag_value + 260,
              y0: 0,
              x1: drag_value + 260,
              y1: 200,
              line: {
                color: 'rgb(55, 128, 191)',
                width: 3
              }
            },
            {
              type: 'vline',
              x0: drag_value + 380,
              y0: 0,
              x1: drag_value + 380,
              y1: 200,
              line: {
                color: 'rgb(55, 128, 191)',
                width: 3
              }
            },
          ]
        };
        
        return Object.assign({}, figure, {
            'layout': layout
            }        
        )
    }
    """,
    Output('graph', 'figure'),
    Input('figure-base', 'data'),
    Input('lap-slider', 'drag_value')
)


# def read_trace_data(trace_hash):
#     trace_data = cache.cache.get(trace_hash)
#
#     return pandas.read_json(trace_data, orient='split')


def main():
    # cache.init_app(app.server, config=CACHE_CONFIG)
    app.run(debug=True)


if __name__ == '__main__':
    print("hello dash")
    main()
