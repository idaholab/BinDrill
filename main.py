import numpy as np
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
from gevent.pywsgi import WSGIServer
from connectors import (
    SqliteTroglodyteSearchConnector,
    # CompareGraphSearchConnector,
    MilvusTroglodyteSearchConnector,
    SqliteDiscoSearchConnector
)
from dash import dcc, html, State, MATCH, ALL
import dash_bootstrap_components as dbc
import dash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import os
import plotly.graph_objs as go
from plotly.graph_objects import layout
from dash import dash_table
import json
from milvus_testing import timing
from matplotlib import cm
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import MinMaxScaler
import threading
import logging
from flask import Flask
from functools import partial
from enum import Enum
import hashlib
import tomli
from OrientDBWrapper import RestAPIWrapper
from pprint import pprint

dir_path = os.path.dirname(os.path.realpath(__file__))

CONFIG_PATH = os.path.join(dir_path, "config.toml")


if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'rb') as f:
        config = tomli.load(f)

server = Flask(__name__)

if 'orientdb' in config:
    orient_conf = config['orientdb']
    databases = RestAPIWrapper(host=orient_conf['host'], port=orient_conf['port'], user=orient_conf['user'], password=orient_conf['pass']).listDatabases()


if os.path.exists(config['function_groups_file']):
    with open(config['function_groups_file']) as f:
        groups = json.load(f)
else:
    logging.error('Function groups files does not exist')

class ConnectorType(Enum):
    sqlite_disco = "Sqlite @DisCo"
    sqlite_troglodyte = "Sqlite Troglodyte"
    milvus_troglodyte = "Milvus Troglodyte"
    na = "---"


class ConnectionRegistry:
    def __init__(self):
        self.d = {}

    def get(self, _id):
        thread_id = os.getpid()
        if thread_id in self.d:
            if _id in self.d[thread_id]:
                return self.d[thread_id][_id]
        logging.debug(f"con {_id} fell through for thread {thread_id}")
        return None

    def register(self, obj):
        thread_id = os.getpid()
        if thread_id not in self.d:
            self.d[thread_id] = {}

        self.d[thread_id][obj.id] = obj
        logging.debug(f"con {obj.id} on thread {thread_id} registered")


CR = ConnectionRegistry()
width = 120
ignore_attribs = ["asm", "llil", "mlil", "hlil", "vex"]

SETTING_TYPE = "setting"
SETTING_LANG_TYPE = "setting-lang"
SETTING_GRAD_TYPE = "setting-grad-lang"


COLLAPSE_TYPE = "collapse"

SUBITEMS = {}


def gen_setting(
    _type,
    label="",
    _id=None,
    placeholder="",
    options=[],
    persistent=True,
    required=True,
    debounce=True,
    default=None,
    style=None,
    inline=True,
    classname=None,
    setting_type=SETTING_TYPE,
):
    _id = {"type": setting_type, "index": _id}
    if _type == "text":
        content = html.Div(
            [
                label,
                dbc.Input(
                    id=_id,
                    placeholder=placeholder,
                    type="text",
                    debounce=debounce,
                    required=required,
                    persistence=persistent,
                ),
            ]
        )
    if _type == "dropdown":
        content = html.Div(
            [
                label,
                dcc.Dropdown(
                    options=[{"label": item, "value": item} for item in options],
                    value=default,
                    id=_id,
                    persistence=persistent,
                ),
            ],
            style=style,
        )

    if _type == "radio":
        content = html.Div(
            [
                label,
                dcc.RadioItems(
                    options=[{"label": item, "value": item} for item in options],
                    value=default,
                    id=_id,
                    persistence=persistent,
                    labelClassName=classname,
                ),
            ]
        )
    return content


def wrap_w_collapse(content, _id, val):
    SUBITEMS[f"{_id}:{val}"] = html.Div(content, className="settingspanel")
    # return html.Div(id={"type": COLLAPSE_TYPE, "index": _id}


def gen_settings():
    settings = []
    settings.extend(
        [
            gen_setting(
                "dropdown",
                label="Query DB Type",
                options=[ConnectorType.sqlite_disco.value,  ConnectorType.sqlite_troglodyte.value, ConnectorType.na.value],
                default="---",
                _id="input_type",
                style={"marginTop": "1rem"},
            ),
            html.Div(id={"type": COLLAPSE_TYPE, "index": "input_type"}),
        ]
    )
    wrap_w_collapse(
        [
            gen_setting(
                "text",
                label="Input DB",
                _id="sqlite_input_path",
                placeholder="Input Database Path",
            ),
            gen_setting(
                "text",
                label="Binary Name",
                # options=[""],
                # default="",
                _id="input_bin_name",
            ),
            gen_setting(
                "dropdown",
                options=databases,
                label="OrientDB Database",
                _id="input_orientdb_database",
                required=False,
                default="---",
            ),
        ],
        "input_type",
        ConnectorType.sqlite_disco.value,
    )
    wrap_w_collapse(
        [
            gen_setting(
                "text",
                label="Input DB",
                _id="sqlite_input_path",
                placeholder="Input Database Path",
            ),
            gen_setting(
                "text",
                label="Binary Name",
                # options=[""],
                # default="",
                _id="input_bin_name",
            ),
        ],
        "input_type",
        ConnectorType.sqlite_troglodyte.value,
    )

    settings.extend(
        [
            gen_setting(
                "dropdown",
                label="Compare DB",
                options=[ConnectorType.sqlite_disco.value, ConnectorType.sqlite_troglodyte.value, ConnectorType.milvus_troglodyte.value, ConnectorType.na.value],
                default="---",
                _id="compare_type",
            ),
            html.Div(id={"type": COLLAPSE_TYPE, "index": "compare_type"}),
        ]
    )

    wrap_w_collapse(
        # gen_setting('text', label='Comparision DB', _id='input_comparision_path', placeholder="Comparision Database Path"),
        [
            gen_setting(
                "text",
                label="Comparision Filter",
                _id="milvus_filter",
                placeholder="",
                required=False,
            ),
            # gen_setting(
            #     "text", label="Input DB", _id="milvus_compare_path", placeholder=""
            # ),
        ],
        "compare_type",
        ConnectorType.milvus_troglodyte.value,
    )

    wrap_w_collapse(
        [
            gen_setting(
                "text",
                label="Comparision DB",
                _id="sqlite_compare_path",
                placeholder="Comparision Database Path",
            ),
            gen_setting(
                "text",
                label="Comparision Filter",
                _id="sqlite_compare_filter",
                required=False,

                placeholder="",
            ),
            gen_setting(
                "dropdown",
                label="OrientDB Database",
                options=databases,
                _id="compare_orientdb_database",
                required=False,
                default="---",
            ),
        ],
        "compare_type",
        ConnectorType.sqlite_disco.value,
    )
    wrap_w_collapse(
        [
            gen_setting(
                "text",
                label="Comparision DB",
                _id="sqlite_compare_path",
                placeholder="Comparision Database Path",
            ),
            gen_setting(
                "text",
                label="Comparision Filter",
                _id="sqlite_compare_filter",
                required=False,
                placeholder="",
            ),
        ],
        "compare_type",
        ConnectorType.sqlite_troglodyte.value,
    )

    return settings


# settings
# input type - milvusconnector
# field, name, type - host, dropdown
#


fig = go.Figure(
    layout={
        "autosize": True,
        "yaxis": {"showticklabels": False},
        "xaxis": {"showticklabels": False},
    }
)

app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True)

app.layout = dbc.Container(
    [
        dcc.Loading(
            id="loading-1",
            type="circle",
            style={
                "position": "fixed",
                "height": "20%",
                "width": "20%",
                "top": "40%",
                "right": "40%",
                "zIndex": 100,
            },
            # className = "position-absolute bottom-50 start-50 translate-middle",
            children=[
                dcc.Store(id="intermediate", storage_type="memory"),
            ],
        ),
        dcc.Store(id="settings-data", storage_type="memory"),
        dcc.Store(id="settings-lang-data", storage_type="memory"),
        dcc.Store(id="settings-grad-data", storage_type="memory"),
        dbc.Button(
            "V",
            id="collapse-button",
            color="primary",
            n_clicks=0,
            className="settingsbutton mb-3",
        ),
        dbc.Collapse(
            dbc.Row(
                gen_settings(),
                id="settings-row",
            ),
            id="collapse",
            is_open=False,
            class_name="settingspanelcollapse",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Col(
                                gen_setting(
                                    "radio",
                                    _id="langtype",
                                    options=["hlil", "asm", "vex"],
                                    default="hlil",
                                    classname="ilbutton",
                                    setting_type=SETTING_LANG_TYPE,
                                )
                            )
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Code(id="code_1", className="codeblock"),
                                class_name="codecol",
                            ),
                            class_name="card cardcodeblock",
                        ),
                        dbc.Row(
                            dbc.Col(
                                html.Code(id="code_2", className="codeblock"),
                                class_name="codecol",
                            ),
                            class_name="card cardcodeblock",
                        ),
                    ],
                    width=3,
                    class_name="codeviews",
                ),
                dbc.Col(
                    [
                        gen_setting(
                            "radio",
                            _id="gradtype",
                            options=["distinct", "score", "function group"],
                            default="score",
                            classname="ilbutton",
                            setting_type=SETTING_GRAD_TYPE,
                        ),
                        dbc.Container(dcc.Graph(figure=fig, id="bin_fig"), fluid=False),
                    ],
                    width=9,
                    class_name="bingraph"
                    # style={"height": "90vh"},
                ),
            ],
            class_name="firstrow",
        ),
        dbc.Row(
            dbc.Col(id="bin_match_table", class_name="matchtable"),
            align="end",
        ),
    ],
    fluid=True,
)

# def to_settings_dict(settings):
#     if isinstance(settings, str):
#         return None
#     settings_dict = {}
#     for item in settings:
#         settings_dict[item.id if hasattr(item, 'id') else f'{item.html_for}_label' ] = to_settings_dict(item.children) if hasattr(item, 'children') else None
#     return settings_dict


def to_settings_dict(settings):
    if isinstance(settings, str):
        return None
    settings_dict = {}
    for item in settings:
        if isinstance(item, dict) and "props" in item:
            item = item["props"]
        settings_dict[item["id"] if "id" in item else f'{item["html_for"]}_label'] = (
            to_settings_dict(item["children"]) if "children" in item else None
        )
    return settings_dict


@app.callback(
    Output("settings-data", "data"),
    Input({"type": SETTING_TYPE, "index": ALL}, "value"),
)
def parse_settings(_):
    return {
        item["id"]["index"]: item["value"] if "value" in item else ""
        for item in dash.callback_context.inputs_list[0]
    }


@app.callback(
    Output("compare_orientdb_database", "value"),
    Output("input_orientdb_database", "value"),
    Input("settings-data", "data"),
)
def orientdb_db_listing(settings_data):
    print(settings_data)
    return "---", "---"


@app.callback(
    Output("settings-lang-data", "data"),
    Input({"type": SETTING_LANG_TYPE, "index": ALL}, "value"),
)
def parse_settings(_):
    return {
        item["id"]["index"]: item["value"] if "value" in item else ""
        for item in dash.callback_context.inputs_list[0]
    }


@app.callback(
    Output("settings-grad-data", "data"),
    Input({"type": SETTING_GRAD_TYPE, "index": ALL}, "value"),
)
def parse_settings(_):
    return {
        item["id"]["index"]: item["value"] if "value" in item else ""
        for item in dash.callback_context.inputs_list[0]
    }


@app.callback(
    Output({"type": COLLAPSE_TYPE, "index": MATCH}, "children"),
    Input({"type": SETTING_TYPE, "index": MATCH}, "value"),
)
def display_subelement(value):
    inputs = dash.callback_context.inputs_list[0]
    val = inputs["value"]
    _id = inputs["id"]["index"]

    if val is None:
        return html.Div()
    else:
        if f"{_id}:{val}" in SUBITEMS:
            return SUBITEMS[f"{_id}:{val}"]
        else:
            return html.Div()


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


###########################
@app.callback(
    Output("code_1", "children"),
    Output("code_2", "children"),
    Output("bin_match_table", "children"),
    Input("bin_fig", "clickData"),
    Input("intermediate", "data"),
    Input("settings-lang-data", "data"),
)
def gen_match_table(bin_fig, intermediate, settings):
    lang_type = settings["langtype"]
    if bin_fig is None:
        return ["", "", None]
    query_func, matched_funcs = intermediate[bin_fig["points"][0]["curveNumber"]]
    matched_funcs.insert(0, query_func)
    code_1_text = (
        query_func[lang_type].strip()
        if (lang_type in query_func and query_func[lang_type] is not None)
        else ""
    )
    code_2_text = (
        matched_funcs[1][lang_type].strip()
        if (lang_type in matched_funcs[1] and matched_funcs[1][lang_type] is not None)
        else ""
    )
    return (
        code_1_text,
        code_2_text,
        dash_table.DataTable(
            id="table",
            # item 1 ought to always be the first "matched item", this is needed as the "query item" does not have score
            columns=[
                {"name": i, "id": i}
                for i in filter(
                    lambda x: x not in ignore_attribs, matched_funcs[1].keys()
                )
            ],
            data=matched_funcs,
            style_data_conditional=[
                {
                    "if": {
                        "row_index": 0,
                    },
                    "fontWeight": "bold",
                },
            ],
            cell_selectable=False,
        ),
    )


def score_gradient(scaler, colorscale, score):
    return "rgb" + str(
        tuple(
            [
                i * 255
                for i in getattr(cm, colorscale)(
                    max([scaler.transform([[score]])[0][0], 0])
                )
            ]
        )
    )


def gen_gradient(**kwargs):
    if "scaler" in kwargs and "colorscale" in kwargs:
        return score_gradient(kwargs["scaler"], kwargs["colorscale"], kwargs["score"])
    elif "color_list" in kwargs:
        return "rgb" + str(next(kwargs["color_list"]))

def group_color_mapping(intermediate, color_list):
    logging.debug('started groups')
    if os.path.exists('func_groups.json'):
        l = []
        func_groups = []
        d = {}
        for i in intermediate:
            func = i[0]
            m = hashlib.md5()
            m.update((func['executable'] + str(func['address'])).encode())
            h_digest = str(m.hexdigest())
            if h_digest in groups:
                func_group = groups[h_digest]
                func_groups.append(func_group)
                if not func_group in d:
                    d[func_group] = next(color_list)
                l.append(d[func_group])
            else:
                l.append('(1.0, 1.0, 1.0)')
                func_groups.append("")
        color_func = partial(gen_gradient, color_list=iter(l)) 
    else:
        color_func = partial(gen_gradient, color_list=color_list) 
    logging.debug('finished groups')
    return func_groups, color_func


@app.callback(
    Output("bin_fig", "figure"),
    Input("intermediate", "data"),
    Input("settings-grad-data", "data"),
)
def gen_fig(intermediate, settings):
    grad_scheme = settings["gradtype"]
    labels = []
    fig = go.Figure()

    if intermediate is None:
        return fig
    func_groups = None
    if grad_scheme == "distinct":
        color_list = cycle(cm.get_cmap("Pastel1").colors)
        color_func = partial(gen_gradient, color_list=color_list)
    elif grad_scheme == "function group":
        color_list = cycle(cm.get_cmap("Pastel1").colors)
        func_groups, color_func = group_color_mapping(intermediate, color_list)
    elif grad_scheme == "score":
        mms = MinMaxScaler()
        mms.fit([[i] for i in config['gradient_range']])
        color_func = partial(gen_gradient, scaler=mms, colorscale="plasma")

        
    base_addr = intermediate[0][0]["address"]
    end_addr = intermediate[-1][0]["address"] + intermediate[-1][0]["bytes"]
    for index, funcs in enumerate(intermediate):
        query_func, matched_funcs = funcs
        top_score = matched_funcs[0]["score"]
        addr, _bytes = query_func["address"], query_func["bytes"]

        start_x = (addr - base_addr) % width
        end_x = (addr + _bytes - base_addr) % width
        start_y = int((addr - base_addr) / width)
        end_y = int((addr + _bytes - base_addr) / width)
        # we need to split/make more complex polygons
        # print(max([mms.transform([[top_score]])[0][0], 0]))
        # print("rgb" + str(cm.plasma(max([mms.transform([[top_score]])[0][0], 0]))))
        if start_y != end_y:
            point_x=[start_x, start_x, width, width, end_x, end_x, 0, 0, start_x]
            point_y=[
                start_y + 1,
                start_y,
                start_y,
                end_y,
                end_y,
                end_y + 1,
                end_y + 1,
                start_y + 1,
                start_y + 1,
            ]
            if end_y - start_y == 1:
                if (width - start_x) > (_bytes/2):
                    label_x =[(start_x + width) /2]
                    label_y=[start_y +.5]
                else:
                    label_x =[end_x /2]
                    label_y=[end_y +.5]
            else:
                label_x =[width/2]
                label_y=[(start_y + end_y+1) /2]
        else:
            point_x=[start_x, start_x, end_x, end_x, start_x]
            point_y=[start_y, start_y + 1, end_y + 1, end_y, start_y]
            label_x =[(start_x + end_x+1) /2]
            label_y=[(start_y + end_y+1) /2]

        fig.add_trace(
            go.Scatter(
                x=point_x,
                y=point_y,
                fill="toself",
                name=query_func["function"][:35],
                mode="lines",
                line_color="black",
                fillcolor=color_func(score=top_score),
                text=f'{"" + func_groups[index] if func_groups is not None else ""}',
                # hovertemplate="<br>test<br>%{text}",

                # fillcolor='rgb'+str(next(colors)),
                customdata=(top_score,),
            ),

        )
        if func_groups is not None and not func_groups[index] is None and func_groups[index] != "" and func_groups[index] != " ":
            label_text = f'{"" + func_groups[index] if func_groups is not None else ""}'
            if (end_x - start_x) < len(label_text):
                label_text = label_text[:_bytes] + '..'
            labels.append(
                go.Scatter(
                    x = label_x,
                    y = [label_y[0]+.05],
                    mode="text",
                    text=label_text,
                    hoverinfo='skip',
                    # hovertemplate="<br>test<br>%{text}",

                    # fillcolor='rgb'+str(next(colors)),
                ),

            )

        # else:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=[start_x, start_x, end_x, end_x, start_x],
        #             y=[start_y, start_y + 1, end_y + 1, end_y, start_y],
        #             fill="toself",
        #             name=query_func["function"],
        #             # fillcolor='rgb'+str(next(colors)),
        #             fillcolor=color_func(score=top_score),
        #             # colorscale="viridis",
        #             mode="lines",
        #             line_color="black",
        #             text=f'{"" + func_groups[index] if func_groups is not None else ""}',
        #             hovertemplate="<br>test<br>%{text}",
        #             # hovertext=[f'name: {query_func["function"]}\n{"" + func_groups[index] if func_groups is not None else ""}'],
        #             # hoverinfo="text",
        #             # size=.5,
        #         ),

        #     )

    fig.update_layout(
        # width=1500,
        height=(end_addr - base_addr) / 7,
        margin=dict(l=0, r=10, b=250, t=15, pad=0),
        # automargin=True,
        # yaxis={"showticklabels": False},
        yaxis={"range": (end_y, 0), "fixedrange": True},
        # xaxis={"showticklabels": False},
        xaxis={"range": (0, 120), "fixedrange": True},
    )

    if grad_scheme == "score":
      
        fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        colorscale='plasma',
                        showscale=True,
                        cmin=0,
                        cmax=1,
                       
                        colorbar=dict(
                            thickness=10,
                            tickvals=[.1, 1],
                            ticktext=["Low Sim", "High Sim"],
                            outlinewidth=1,
                            # orientation='h',
                            yanchor= 'top',
                            x=-.13,
                            y=1,
                            xpad = 50,
                            ypad = 0,
                            lenmode = 'pixels',
                            len = 200,
                            ticklabelposition = 'outside bottom',
                            tickcolor='black',
                            tickfont={'color': 'black'}
                        ),
                    ),
                    hoverinfo="none",
                )
            )

    for label in labels:
        fig.add_trace(label)

    return fig


# @app.callback(Output("input_bin_name", "options"), Input("settings-data", "data"), prevent_initial_call=True)
# def get_bins(settings):
#     if "sqlite_input_path" not in settings or "input_bin_name" not in settings:
#         raise PreventUpdate
#     print('hit')
#     input_bin_path  = settings["sqlite_input_path"]
#     # bin_con = CompareGraphSearchConnector(path=input_bin_path)
#     bin_con = SqliteMilvusSearchConnector(path=input_bin_path)

#     bins = bin_con.getbins()
#     bins = [{"label": item[0], "value": item[0]} for item in bins]
#     return bins


@app.callback(
    Output("intermediate", "data"),
    Input("settings-data", "data"),
    # Input("input_bin_name", "value"),
    # Input("input_comparision_path", "value"),
    # Input("input_comparision_filter", "value"),
)
@timing
def get_inputs(settings):
    print(dash.callback_context.triggered[0])
    for input_text in ["sqlite_input_path", "input_bin_name"]:
        if input_text not in settings or settings[input_text] == "":
            return None

    if not os.path.exists(settings["sqlite_input_path"]):
        raise Exception("Input binary database path incorrect")
    # if not os.path.exists(input_comparision_path):
    #     raise Exception("Input comparision database path incorrect")

    # bin_con = CompareGraphSearchConnector(path=input_bin_path)
    # comp_con = CompareGraphSearchConnector(
    #     path=input_comparision_path, func_filter=comp_filter
    # )
    d = {
        ConnectorType.milvus_troglodyte.value: MilvusTroglodyteSearchConnector,
        ConnectorType.sqlite_disco.value: SqliteDiscoSearchConnector,
        ConnectorType.sqlite_troglodyte.value: SqliteTroglodyteSearchConnector,
    }
    if settings['input_type'] not in d:
        return None
    else:
        bin_connector = d[settings['input_type']]

    compare_args = {}
    if 'compare_orientdb_database' in settings:
        compare_args['orientdb_db'] = settings['compare_orientdb_database']

    input_args = {}
    if 'input_orientdb_database' in settings:
        input_args['orientdb_db'] = settings['input_orientdb_database']

    pprint(settings)
    bin_con = CR.get(
        bin_connector.gen_id(sqlite_path=settings["sqlite_input_path"])
    )
    if bin_con is None:
        bin_con = bin_connector(config=config, **input_args)
        bin_con.connect(connect=False)
        bin_con.set_input_sqlite(settings["sqlite_input_path"])
        CR.register(bin_con)
    if settings['compare_type'] not in d:
        return None
    else:
        comp_connector = d[settings['compare_type']]
    comp_con = CR.get(comp_connector.gen_id(MC=True))
    if comp_con is None:
        comp_con = comp_connector(config=config, **compare_args)
        if isinstance(comp_con, MilvusTroglodyteSearchConnector):
            comp_con.connect()
        else:
            comp_con.connect(sqlite_file=settings["sqlite_compare_path"])
        CR.register(comp_con)

    # this should return the function details in the same order as the submitted embeddings
    gotten_funcs = bin_con.getfuncs(include_bin=settings["input_bin_name"], limit=config['func_limit'])
    print('func num', len(gotten_funcs))
    if settings['input_type'] == ConnectorType.milvus_troglodyte.value:
        func_filter=settings["milvus_filter"] if "milvus_filter" in settings else ""
    else:
        func_filter=settings["sqlite_compare_filter"] if "sqlite_compare_filter" in settings else ""
    return comp_con.search(
        gotten_funcs,
        func_filter=func_filter,
        exclude_bin=settings["input_bin_name"],
    )


######################
if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    app.run_server(debug=True, host=config['binarydriller_host'], port=config['binarydriller_port'])
    # http_server = WSGIServer((config['binarydriller_host'], config['binarydriller_port']), server)
    # http_server.serve_forever()
