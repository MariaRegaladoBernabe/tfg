import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash_slicer import VolumeSlicer
import dash_table
import plotly.graph_objs as go

import numpy as np
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

import plotly.graph_objects as go
import plotly.express as px

# from skimage import io
from skimage.color import rgb2gray
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skimage as sk
from skimage import data, filters, exposure, morphology, img_as_ubyte, segmentation
from skimage.util import compare_images
from skimage.morphology import erosion, dilation, disk, square, closing, rectangle, diameter_closing
import skimage.filters as flt
from skimage.filters import median, threshold_local, threshold_otsu, roberts, sobel, gaussian, rank
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import active_contour, chan_vese
from skimage.draw import polygon
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1

from scipy.signal.signaltools import wiener
from scipy import ndimage

import cv2 as cv
import numpy as np

from PIL import Image
import json

import re
import time
import os

import base64
import datetime
import pandas as pd

import io


#===============================================================================
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#===============================================================================

DEBUG = True

NUM_ATYPES = 10
DEFAULT_FIG_MODE = "layout"
annotation_colormap = px.colors.qualitative.Light24
annotation_types = [
    "Cingulate Sulcus",
    "Central Sulcus",
    "Precentral Sulcus",
    "Postcentral Sulcus",
    "Sylvian Fissure",
    "Superior Temporal Sulcus",
    "Temporo-Occipital Sulcus",
    "Parieto-Occipital Sulcus",
    "Calcarine Fissure",
    "Superior Frontal Sulcus"
]
DEFAULT_ATYPE = annotation_types[0]

# prepare bijective type<->color mapping
typ_col_pairs = [
    (t, annotation_colormap[n % len(annotation_colormap)])
    for n, t in enumerate(annotation_types)
]
# types to colors
color_dict = {}
# colors to types
type_dict = {}
for typ, col in typ_col_pairs:
    color_dict[typ] = col
    type_dict[col] = typ

options = list(color_dict.keys())
columns = ["Type", "Coordinates"]

# # Open the readme for use in the context info
# with open("assets/Howto.md", "r") as f:
#     # Using .read rather than .readlines because dcc.Markdown
#     # joins list of strings with newline characters
#     howto = f.read()


def debug_print(*args):
    if DEBUG:
        print('debug print:')
        print(*args)


def coord_to_tab_column(coord):
    return coord.upper()


def time_passed(start=0):
    return round(time.mktime(time.localtime())) - start


def format_float(f):
    return "%.2f" % (float(f),)


def shape_to_table_row(sh):
    return {
        "Type": type_dict[sh["line"]["color"]],
        "Coordinates": sh["path"]
    }

def table_row_to_shape(tr):
    return {
        "editable": True,
        "xref": "x",
        "yref": "y",
        "layer": "above",
        "opacity": 1,
        "line": {"color": color_dict[tr["Type"]], "width": 2, "dash": "solid"},
        "fillcolor": "rgba(0, 0, 0, 0)",
        "fillrule": "evenodd",
        "type": "path",
        "path": tr["Coordinates"],
    }


def shape_cmp(s0, s1):
    """ Compare two shapes """
    return (
        (s0["path"] == s1["path"])
        and (s0["line"]["color"] == s1["line"]["color"])
    )


def shape_in(se):
    """ check if a shape is in list (done this way to use custom compare) """
    return lambda s: any(shape_cmp(s, s_) for s_ in se)

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


def index_of_shape(shapes, shape):
    for i, shapes_item in enumerate(shapes):
        if shape_cmp(shapes_item, shape):
            return i
    raise ValueError  # not found


def annotations_table_shape_resize(annotations_table_data, fig_data):
    """
    Extract the shape that was resized (its index) and store the resized
    coordinates.
    """

    index = int(((list(fig_data.keys())[0]).split('[')[1]).split(']')[0])
    annotations_table_data[index]['Coordinates'] = fig_data['shapes[' + str(index) + '].path']

    return annotations_table_data


def shape_data_remove_timestamp(shape):
    """
    go.Figure complains if we include the 'timestamp' key when updating the
    figure
    """
    new_shape = dict()
    for k in shape.keys() - set(["timestamp"]):
        new_shape[k] = shape[k]
    return new_shape


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print('EXCEPTION: ', e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_dict('records')


#Function to create a dictionary
def dictionary_function(vector):
    vector_dictionary = {}
    for i in range(len(vector)):
        id = vector[i]
        vector_dictionary[id] = i
    return vector_dictionary

# Function to define the path of each image in the Data Base
def definition_filelist(dictionary):

    file_list_all = []
    filelist_cont = []

    for baby_id, baby_weeks in dictionary.items():
        file_list_baby = []
        for week_id, baby_planes in baby_weeks.items():
            file_list_week = []
            for plane_id in baby_planes:
                filename = path + baby_id+'/' + baby_id+'_'+week_id+'/' + baby_id+'_'+week_id+'_'+plane_id+'.jpg'
                file_list_week.append(filename)
                filelist_cont.append(filename)
            #file_list_week.append(path+i+"/"+i+"_"+ii+"EG_SELECCION ESTANDAR/"+i+"_"+iii+"_M_CUT.jpg")
            file_list_baby.append(file_list_week)
        file_list_all.append(file_list_baby)

    return file_list_all, filelist_cont

# Function to obtain get the babies
def get_subfolders(dirname):
    items = []

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for i in subfolders:
        item = i.split('/')[-1]
        items.append(item)

    return sorted(items)

# Function to obtain get the planes
def get_planes(dirname):
    planes = []
    files = [file for file in os.listdir(dirname) if file.endswith('.jpg')]
    for f in files:
        item = f.split("_")
        temp = item[-1]
        temp = temp.split(".")

        planes.append(temp[0])

    return sorted(planes)

def get_babies_weeks_planes(babies_list):

    babies_dictionary = {}
    weeks_list = []
    for b in babies_list:
        path_week = path + '/' + b
        weeks_path = get_subfolders(path_week)

        #Values
        for w in weeks_path:
            item = w.split("_")
            weeks_list.append(item[1])

        #Add weeks to the weeks dictionary
        weeks_dictionary = dictionary_function(weeks_list)

        # ----------------- STEP 3: Create planes dictionary ------------
        for w in weeks_list:
            path_planes = path_week + '/' + b + "_" + w
            planes = get_planes(path_planes)
            weeks_dictionary[w] = planes

        #Add weeks to the babies dictionary
        babies_dictionary[b] = weeks_dictionary

        #Reset weeks dictionary
        weeks_list = []
        weeks_dictionary = {}

    return babies_dictionary

path = "./Data_Base_update/"

# Babies
babies_list = get_subfolders(path)
babies_indexes = dictionary_function(babies_list)
baby_default = babies_list[0]

# Babies, weeks and planes
babies_dictionary = get_babies_weeks_planes(babies_list)

# Weeks
weeks = list(babies_dictionary[baby_default].keys())
weeks_indexes = dictionary_function(weeks)
week_default = weeks[0]

# Planes
planes = babies_dictionary[baby_default][week_default]
planes_indexes = dictionary_function(planes)
plane_default = planes[0]

# Files paths
filelist_path_agrup, filelist_path_cont = definition_filelist(babies_dictionary)

# Previous selected variables
image_id_baby_before = -1
image_id_week_before = -1
image_id_plane_before = -1

# Default image path
index_baby = babies_indexes[baby_default]
index_week = weeks_indexes[week_default]
index_plane = planes_indexes[plane_default]
path_img_ini = filelist_path_agrup[index_baby][index_week][index_plane]

# Show image
fig = px.imshow(sk.io.imread(path_img_ini))
# if (os.path.exists(path_img_ini) == True):
#     fig = px.imshow(io.imread(path_img_ini))
# else:
#     img = np.ones((550,868))
#     fig = px.imshow(img, binary_string=True)

fig.update_layout(
    newshape_line_color=color_dict[DEFAULT_ATYPE],
    margin=dict(l=0, r=0, b=0, t=0, pad=4),
    dragmode="drawclosedpath",
    uirevision=True
)


# Deffinition Buttons
button_babies = dcc.Dropdown(
    id='babies_button',
    options=[{'label': i, 'value': i} for i in babies_list],
    value=baby_default,
    clearable=False
),

button_weeks = dcc.Dropdown(
    id='weeks_button',
    options=[{'label': i, 'value': i} for i in weeks],
    value=week_default,
    clearable=False
),

button_planes = dcc.Dropdown(
    id='planes_button',
    options=[{'label': i, 'value': i} for i in planes],
    value=plane_default,
    clearable=False
),

# + Buttons

button_howto = dbc.Button(
    "View Code on github",
    outline=True,
    color="primary",
    href="https://github.com/Derther/app_TFM",
    id="gh-link",
    style={"text-transform": "none"},
)


# ------------- Define App Layout ---------------------------------------------------

Image_annotation_card_Doctors = dbc.Card(
    id = 'image_card_doctors',
    children = [
        dbc.CardHeader(html.H5(id='tittle_card_Doctors')),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'Image_baby_week_plane',
                    figure = fig,
                    config = {
                        'scrollZoom' : True,
                        'modeBarButtonsToAdd': [
                            'drawclosedpath',
                            'eraseshape'
                        ],
                    },
                    responsive = True,
                ),

            ]
        ),
    ]
),


plot_fig_segmented = dbc.Card(
    id = 'image_segmented_post',
    children = [
        dbc.CardHeader(html.H5('Image Segmented')),
        dbc.CardBody(
            children = [
                dcc.Graph(
                    id = 'image_to_segment',
                    figure = fig,
                    config = {
                        'scrollZoom' : True,
                        'modeBarButtonsToAdd': [
                            'drawclosedpath',
                            'eraseshape'
                        ],
                    },
                    responsive = True,
                ),

            ]
        ),
    ]
),


# The button for the layout is saved as a variable
run_segmentation_button = html.Div(
    children = [
        dbc.Button("Run segmentation", id="run_segmentation_button",  color="primary"),
        html.Span(id="run_segmentation_cont", style={"vertical-align": "middle"}),
    ],
)

#####################################################################################

options_segmentation = ['Threshold', 'Sigmoid + Threshold', 'Snake']

button_options_segmentation = dcc.Dropdown(
    id='button_options_segmentation',
    options=[{'label': i, 'value': i} for i in options_segmentation],
    value=options_segmentation[0],
    clearable=False
),
#####################################################################################

button_import = dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'width': '80%',
        'height': '50px',
        'lineHeight': '50px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '2px',
        'textAlign': 'center',
        'margin': '5px'
    },
    # Not allow multiple files to be uploaded
    multiple=False
),


button_import_data = html.Div(
    [
        dbc.Row(
            button_import,
        ),
    ]
)


Buttons_select_children = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    button_babies,
                ),
                dbc.Col(
                    button_weeks,
                ),
                dbc.Col(
                    button_planes,
                ),
#                 dbc.Col(
#                     run_segmentation_button,
#                 ),
            ],

        ),
    ]
)


Button_segmentation = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    run_segmentation_button,
                ),
                dbc.Col(
                    button_options_segmentation,
                ),
            ]
        ),
    ]
)


annotated_data_card_Doctor= dbc.Card(
    [
        dbc.CardHeader(dbc.Row(dbc.Col(html.H5("Annotated data")))),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H5("Coordinates of annotations"))),
                dbc.Row(
                    dbc.Col(
                        [
                            dash_table.DataTable(
                                id="annotations-table",
                                columns=[
                                    dict(
                                        name=n,
                                        id=n,
                                        presentation=(
                                            "dropdown" if n == "Type" else "input"
                                        ),
                                    )
                                    for n in columns
                                ],
                                editable=True,
                                style_data={"height": 40},
                                style_table={'height': '150px', 'overflowY': 'auto'},
                                style_cell={
                                    "overflow": "hidden",
                                    "textOverflow": "ellipsis",
                                    "maxWidth": 0,
                                },
                                dropdown={
                                    "Type": {
                                        "options": [
                                            {"label": o, "value": o}
                                            for o in annotation_types
                                        ],
                                        "clearable": False,
                                    }
                                },
                                style_cell_conditional=[
                                    {"if": {"column_id": "Type"}, "textAlign": "left"},
                                ],

                                fill_width=True,
                                row_deletable=True,
                                export_format='csv',
                                export_headers='display',
                            ),
                            dcc.Store(id="graph-copy", data=fig),
                            dcc.Store(
                                id="annotations-store",
                                data=dict(
                                    **{
                                        filename: {"shapes": []}
                                        for filename in filelist_path_cont
                                    },
                                    **{"starttime": time_passed()}
                                ),
                            ),
                            dcc.Store(
                                id="image_files",
                                data={"files": filelist_path_agrup, "current_baby": 0, "current_week":0, "current_plane": 0},
                            ),
                        ],
                    ),
                ),

            ]
        ),
        dbc.CardFooter(
            children = [
                html.Div(
                    dbc.Row(
                        dbc.Col(
                            [
                                html.H5("Create new annotation for"),
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="annotation-type-dropdown",
                                            options=[
                                                {"label": t, "value": t} for t in annotation_types
                                            ],
                                            value=DEFAULT_ATYPE,
                                            clearable=False,
                                        ),
                                    ),
                                ]),
                            ],
                            align="center",
                        ),
                    ),
                ),
            ],
        ),
    ],
)

# The name of the table columns is defined
# dic_names_columns = [{'name': 'Type', 'id': 'column_1'}, {'name': 'Coordinates', 'id': 'column_2'}]
# lista_rows = []
# dic_data_table = {'column_1': 'Empty', 'column_2': 'Empty'}
# lista_rows.append(dic_data_table)
dic_names_columns = [{'name': 'Type', 'id': 'Type'}, {'name': 'Coordinates', 'id': 'Coordinates'}]
lista_rows = []
dic_data_table = {'Type': 'Empty', 'Coordinates': 'Empty'}
lista_rows.append(dic_data_table)

# The segmented table for the layout is saved as a variable
table_DataTable = dash_table.DataTable(
    id='table_segmentation',
    columns=dic_names_columns,
    data=lista_rows,
    style_table={'height': '150px', 'overflowY': 'auto'},
    style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'maxWidth': 0
    },
    style_cell_conditional=[
        {"if": {"column_id": "Type"}, "textAlign": "left"}
    ],
    export_format='csv',
    export_headers='display',

)

#The card of the segmented table is defined
table_coordinates_segmentation = dbc.Card(
    children = [
        dbc.CardHeader(html.H5('Table coordinates segmentation')),
        dbc.CardBody(
            [
                dbc.Row(dbc.Col(html.H5('Coordinates of segmentation'))),
                dbc.Row(
                    dbc.Col(
                        table_DataTable,
                    )
                )
            ]
        )
    ]
)



# Define Modal
with open("assets/modal.md", "r") as f:
    howto_md = f.read()

modal_overlay = dbc.Modal(
    [
        dbc.ModalBody(html.Div([dcc.Markdown(howto_md)], id="howto-md")),
        dbc.ModalFooter(dbc.Button("Close", id="howto-close", className="howto-bn")),
    ],
    id="modal",
    size="lg",
)


nav_bar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.A(
                                        html.Img(
                                            src=app.get_asset_url("dash-logo-new.png"),
                                            height="30px",
                                        ),
                                        href="https://plotly.com/dash/",
                                    ),
                                    style={"width": "min-content"},
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H3("Sant Joan de Déu app"),
                                            html.P(
                                                "Groove segmentation"
                                            ),
                                        ],
                                        id="app_title",
                                    )
                                ),
                            ],
                            align="center",
                            style={"display": "inline-flex"},
                        )
                    ),
                    dbc.Col(
                        [
                            dbc.NavbarToggler(id="navbar-toggler"),
                            dbc.Collapse(
                                dbc.Nav(
                                    [dbc.NavItem(button_howto)],
                                    className="ml-auto",
                                    navbar=True,
                                ),
                                id="navbar-collapse",
                                navbar=True,
                            ),
                        ]
                    ),
                    modal_overlay,
                ],
                align="center",
                style={"width": "100%"},
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)


app.layout = html.Div(
    [
        nav_bar,
        dbc.Container(
            children = [
                dbc.Row([dbc.Col(Buttons_select_children, md = 6), dbc.Col(Button_segmentation)]),
                dbc.Row(dbc.Col(button_import_data, width={"size": 5, "offset": 7})),
                dbc.Row([dbc.Col(Image_annotation_card_Doctors, md = 6), dbc.Col(annotated_data_card_Doctor)]),
                dbc.Row([dbc.Col(plot_fig_segmented, md = 6), dbc.Col(table_coordinates_segmentation)]),
#                 dbc.Row([dbc.Col(Image_annotation_card_Doctors, md = 6), dbc.Col(plot_fig_segmented, md = 6)]),
#                 dbc.Row([dbc.Col(annotated_data_card_Doctor, md = 6), dbc.Col(table_coordinates_segmentation, md = 6)]),
            ],
            fluid=True,
        ),
        # dcc.Store(id="annotations", data={}),
        # dcc.Store(id="occlusion-surface", data={}),
    ],
)


# ------------- Define App Interactivity ---------------------------------------------------


@app.callback(
    Output(component_id="tittle_card_Doctors", component_property="children"),
    [
        Input(component_id="babies_button", component_property="value"),
        Input(component_id="weeks_button", component_property="value"),
        Input(component_id="planes_button", component_property="value"),
    ]
)
def update_output_div(baby, week, plane):
    return 'Groove selection of the Baby {}, week {} and plane {}'.format(baby, week, plane)

#@app.callback(
#    [
#        dash.dependencies.Output('weeks_button', 'options')
#    ],
#    [
#        dash.dependencies.Input('babies_button', 'value'),
#    ]
#)
#def update_weeks_planes_dropdown(baby_selected):
#    #return ([{'label': i, 'value': i} for i in fnameDict[name]])
#    weeks = list(babies_dictionary[baby_selected].keys())
#    week_options=[{'label': i, 'value': i} for i in weeks]
#    planes = babies_dictionary[baby_selected][weeks[0]]
#    planes_options=[{'label': i, 'value': i} for i in planes]

#    return ([week_options])


@app.callback(
    [Output("annotations-table", "data"),
     Output("image_files", "data"),
     Output("babies_button", "value"),
     Output("weeks_button", "value"),
     Output("planes_button", "value")],
    [
        Input("babies_button", "value"),
        Input("weeks_button", "value"),
        Input("planes_button", "value"),
        Input("Image_baby_week_plane", "relayoutData"),
        Input('upload-data', 'contents')
    ],
    [
        State("annotations-table", "data"),
        State("image_files", "data"),
        State("annotations-store", "data"),
        State("annotation-type-dropdown", "value"),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified')
    ],
)
def modify_table_entries(
    id_baby,
    id_week,
    id_plane,
    graph_relayoutData,
    list_of_contents,
    annotations_table_data,
    image_files_data,
    annotations_store_data,
    annotation_type,
    list_of_names,
    list_of_dates
):
    global image_id_baby_before, image_id_week_before, image_id_plane_before
    global babies_dictionary, weeks_dictionary, planes_dictionary
    global babies_indexes, weeks_indexes, planes_indexes

    # Callback called because shape has been created or modified
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext == "Image_baby_week_plane.relayoutData":

        # New or modified shape
        if "shapes" in graph_relayoutData.keys():
            # this means all the shapes have been passed to this function via
            # graph_relayoutData, so we store them
            annotations_table_data = [
                shape_to_table_row(sh) for sh in graph_relayoutData["shapes"]
            ]
        elif re.match("shapes\[[0-9]+\].path", list(graph_relayoutData.keys())[0]):
            # this means a shape was updated (e.g., by clicking and dragging its
            # vertices), so we just update the specific shape
            annotations_table_data = annotations_table_shape_resize(
                annotations_table_data, graph_relayoutData
            )

        # Return depending on shapes
        if annotations_table_data is None:
            return dash.no_update
        else:
            return (annotations_table_data, image_files_data, id_baby, id_week, id_plane)

    # Callback called because updat csv file
    elif cbcontext == "upload-data.contents":

        children = []
        if list_of_contents is not None:
            children = parse_contents(list_of_contents, list_of_names, list_of_dates)
        return (children, image_files_data, id_baby, id_week, id_plane)

    # Callback called because button value
    else:

        image_id_baby = babies_indexes[id_baby]
        image_id_week = weeks_indexes[id_week]
        image_id_plane = planes_indexes[id_plane]

        # Update configuration
        image_files_data["current_baby"] = image_id_baby
        image_files_data["current_week"] = image_id_week
        image_files_data["current_plane"] = image_id_plane

        # If some value (baby, week, plane) has changed
        if (image_id_baby != image_id_baby_before or image_id_week != image_id_week_before or image_id_plane != image_id_plane_before):

            # Update variables
            if image_id_baby != image_id_baby_before:

                # Weeks
                weeks = list(babies_dictionary[id_baby].keys())
                weeks_indexes = dictionary_function(weeks)
                if id_week in weeks_indexes:
                    image_id_week = weeks_indexes[id_week]
                else:
                    image_id_week = 0
                    id_week = list(weeks_indexes.keys())[image_id_week]

                # Planes
                planes = babies_dictionary[id_baby][id_week]
                planes_indexes = dictionary_function(planes)
                if id_plane in planes_indexes:
                    image_id_plane = planes_indexes[id_plane]
                else:
                    image_id_plane = 0
                    id_plane = list(planes_indexes.keys())[image_id_plane]

            elif image_id_week != image_id_week_before:

                # Planes
                planes = babies_dictionary[id_baby][id_week]
                planes_indexes = dictionary_function(planes)
                if id_plane in planes_indexes:
                    image_id_plane = planes_indexes[id_plane]
                else:
                    image_id_plane = 0
                    id_plane = list(planes_indexes.keys())[image_id_plane]

            # Update configuration
            image_files_data["current_baby"] = image_id_baby
            image_files_data["current_week"] = image_id_week
            image_files_data["current_plane"] = image_id_plane

            # Update "previous" configuration
            image_id_baby_before = image_files_data["current_baby"]
            image_id_week_before = image_files_data["current_week"]
            image_id_plane_before = image_files_data["current_plane"]

            # Update annotations_table_data with new data
            annotations_table_data = []
            filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]
            for sh in annotations_store_data[filename]["shapes"]:
                annotations_table_data.append(shape_to_table_row(sh))
            return (annotations_table_data, image_files_data, id_baby, id_week, id_plane)
        else:
            # No update anything
            return (annotations_table_data, image_files_data, id_baby, id_week, id_plane)

@app.callback(
    [Output("Image_baby_week_plane", "figure"),
     Output("annotations-store", "data"),
     Output('weeks_button', 'options'),
     Output('planes_button', 'options')
    ],

    [Input("annotations-table", "data"),
     Input("annotation-type-dropdown", "value"),
     Input("babies_button", "value"),
     Input("weeks_button", "value"),
     Input("planes_button", "value")],

    [State("image_files", "data"), State("annotations-store", "data")],
)
def send_figure_to_graph(
    annotations_table_data, annotation_type,
    num_baby, num_week, num_plane,
    image_files_data, annotations_store
):

    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]

    # if cbcontext == "babies_button.value" or cbcontext == "weeks_button.value" or cbcontext == "planes_button.value":
    weeks_list = list(babies_dictionary[num_baby].keys())
    planes_list = babies_dictionary[num_baby][num_week]

    if annotations_table_data is not None:
        # File path (current baby, week and plane updated in the previous callback)
        filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]

        # Convert table rows to those understood by fig.update_layout
        fig_shapes = [table_row_to_shape(sh) for sh in annotations_table_data]

        # Find the shapes that are new
        new_shapes_i = []
        old_shapes_i = []
        for i, sh in enumerate(fig_shapes):
            if not shape_in(annotations_store[filename]["shapes"])(sh):
                new_shapes_i.append(i)
            else:
                old_shapes_i.append(i)

        # Add timestamps to the new shapes
        for i in new_shapes_i:
            fig_shapes[i]["timestamp"] = time_passed(annotations_store["starttime"])

        # Find the old shapes and look up their timestamps
        for i in old_shapes_i:
            old_shape_i = index_of_shape(
                annotations_store[filename]["shapes"], fig_shapes[i]
            )
            fig_shapes[i]["timestamp"] = annotations_store[filename]["shapes"][old_shape_i]["timestamp"]
        shapes = fig_shapes

        # Show image
        fig = px.imshow(sk.io.imread(filename), binary_backend="bmp")
        # if (os.path.exists(filename) == True):
        #     fig = px.imshow(io.imread(filename), binary_backend="bmp")
        # else:
        #     img = np.ones((550,868))
        #     fig = px.imshow(img, binary_string=True)

        fig.update_layout(
            shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
            # reduce space between image and graph edges
            newshape_line_color=color_dict[annotation_type],
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawclosedpath",
            uirevision=True
        )

        # Update table with the annotations
        annotations_store[filename]["shapes"] = shapes

        return (fig, annotations_store, weeks_list, planes_list)
        # return (fig, annotations_store)
    else:
        return dash.no_update


@app.callback(
    [Output(component_id='run_segmentation_cont', component_property='children'),
     Output("image_to_segment", "figure"),
     Output('table_segmentation', 'data')],

    [Input(component_id='run_segmentation_button', component_property='n_clicks'),
     Input("babies_button", "value"),
     Input("weeks_button", "value"),
     Input("planes_button", "value"),
     Input("image_to_segment", "relayoutData"),
     Input("button_options_segmentation", "value")],

    [State("image_files", "data"),
     State("annotations-store", "data"),
     State("table_segmentation", "data")],
)

def update_output(n_clicks, id_baby, id_week, id_plane, image_to_seg, option_segmentation, image_files_data, annotations_store_data, segmentation_store_data):

    # Why this callback has been called
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]

    # Callback called because segmentation has been done and the shape of the
    # segmented image has been modified
    if n_clicks is not None and cbcontext == "image_to_segment.relayoutData":
        # New or modified shape
        if "shapes" in image_to_seg.keys():
            # this means all the shapes have been passed to this function via
            # graph_relayoutData, so we store them
            segmentation_store_data = [shape_to_table_row(sh) for sh in image_to_seg["shapes"]]
        elif re.match("shapes\[[0-9]+\].path", list(image_to_seg.keys())[0]):
            # this means a shape was updated (e.g., by clicking and dragging its
            # vertices), so we just update the specific shape
            segmentation_store_data = annotations_table_shape_resize(segmentation_store_data, image_to_seg)

        # Return depending on shapes
        if segmentation_store_data is None:
            return dash.no_update
        else:
            # Interaction type
            action = list(image_to_seg.keys())[0]
            if action != 'dragmode':

                # Row type
                index = int((action.split('[')[1]).split(']')[0])
                type_line = segmentation_store_data[index]['Type']

                # Shapes of the figure
                shapes = [table_row_to_shape(sh) for sh in segmentation_store_data]

                # File path
                filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]

                # Show segmented image
                img = sk.io.imread(filename, as_gray=True)
                fig = px.imshow(img, binary_string=True)
                fig.update_layout(
                    shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
                    newshape_line_color=color_dict[type_line],
                    margin=dict(l=0, r=0, b=0, t=0, pad=4),
                    dragmode="drawclosedpath",
                    uirevision=True)
                return (" ", fig, segmentation_store_data)
            else:
                return dash.no_update

    # Segmentation has not been done yet
    if n_clicks is None:
        # Black image
        img = np.ones((550,868))
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(
#             newshape_line_color=color_dict[DEFAULT_ATYPE],
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawclosedpath",
            uirevision=True
        )

        return (" ", fig, [])

    # Segmentation has to be done or has been done already
    else:
        # Annotations of the segmented image
        annotation_store_segmented = annotations_store_data

        # File path
        filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]

        # Figure image
        if (os.path.exists(filename) == True):

            # Image
            img = sk.io.imread(filename, as_gray=True)
            fig = px.imshow(img, binary_string=True)
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0, pad=4),
                dragmode="drawclosedpath",
                uirevision=True
            )

            # Shapes drawed in the original image
            coordenadas_grooves_manual = []
            for sh in annotations_store_data[filename]["shapes"]:
                coordenadas_grooves_manual.append(shape_to_table_row(sh))

            # If not drawed anything
            lista_rows = []
            if coordenadas_grooves_manual == []:
                dic_data_table = {'Type': 'Empty', 'Coordinates': 'Empty'}
                lista_rows.append(dic_data_table)

            # If something has been drawed
            else:

                # For each shape, save annotation type and coordinates
                name_coordinates = []; coordinates_segmentation = []
                for t in coordenadas_grooves_manual:
                    name = t["Type"]
                    coor = t["Coordinates"][1:-1].split('L')

                    name_coordinates.append(name); coordinates_segmentation.append(coor)

                # Points coordinates of each annotation
                x_coordinates_each_segmentation = []
                y_coordinates_each_segmentation = []
                for i in range(len(coordenadas_grooves_manual)):

                    # For each coordinate of the annotation shape contour
                    x_contour = []
                    y_contour = []
                    for iii in range(len(coordinates_segmentation[i])):

                        # Point coordinates
                        x_i, y_i = coordinates_segmentation[i][iii].split(",")
                        x_contour.append(float(x_i))
                        y_contour.append(float(y_i))

                        # Save the initial point
                        if (iii==0):
                            x_close = float(x_i)
                            y_close = float(y_i)

                        # Add initial point to close the shape
                        if (iii==(len(coordinates_segmentation[i])-1)):
                            x_contour.append(x_close)
                            y_contour.append(y_close)

                    x_coordinates_each_segmentation.append(np.uint(x_contour))
                    y_coordinates_each_segmentation.append(np.uint(y_contour))

                # For each annotation shape
                shapes = []
                lista_rows_segmented = []
                for tt in range(len(x_coordinates_each_segmentation)):

                    # Coordinates of all the points of the shape
                    x = x_coordinates_each_segmentation[tt]
                    y = y_coordinates_each_segmentation[tt]

                    # Take region of the sulcre (bounding box)
                    x_max, x_min = np.uint(np.max(x)), np.uint(np.min(x))
                    y_max, y_min = np.uint(np.max(y)), np.uint(np.min(y))

###############################################################################################################################################

                    if ((option_segmentation == 'Threshold') or (option_segmentation == 'Sigmoid + Threshold')):

                        if (option_segmentation == 'Threshold'):
                            img_cut = img[y_min:y_max, x_min:x_max]
                            img_cut_preproces = (img_cut-np.min(img_cut))/(np.max(img_cut)-np.min(img_cut))

                        elif (option_segmentation == 'Sigmoid + Threshold'):

                            # Apply Sigmoid Correction to the image
                            sigmoid_image = sk.exposure.adjust_sigmoid(img, 0.4, 10)

                            img_cut = sigmoid_image[y_min:y_max, x_min:x_max]
                            img_cut_preproces = (img_cut-np.min(img_cut))/(np.max(img_cut)-np.min(img_cut))


                        # Threshold Local
                        func = lambda arr: arr.mean()
                        func2 = lambda arr: arr.std()
                        binary_image = (img_cut_preproces > threshold_local(img_cut_preproces, 45, 'generic', param=func))
                        radius = 1
                        selem = disk(radius)
                        closing_image = closing(binary_image, selem)

                        img_treated = closing_image

                    elif (option_segmentation == 'Snake'):
                        # Apply Sigmoid Correction to the image
                        sigmoid_image = sk.exposure.adjust_sigmoid(img, 0.4, 10)

                        img_cut = sigmoid_image[y_min:y_max, x_min:x_max]
                        img_cut_preproces = (img_cut-np.min(img_cut))/(np.max(img_cut)-np.min(img_cut))

                        # Morphological ACWE
                        # Initial level set
                        init_ls = checkerboard_level_set(img_cut_preproces.shape, 2)
                        init_ls2 = checkerboard_level_set(img_cut_preproces.shape,2)
                        # List with intermediate results for plotting the evolution
                        evolution = []
                        callback = store_evolution_in(evolution)
                        ls2 = morphological_chan_vese(img_cut_preproces, num_iter=4, init_level_set=init_ls2,
                                                     smoothing=3, iter_callback=callback)

                        sum_pixel = np.sum(ls2)
                        img_test_pixel = ls2.shape[0]*ls2.shape[1]

                        ones_percentage = sum_pixel / img_test_pixel

                        # Inverts if number of 1s is greater than number of 0s
                        if ones_percentage > 0.5:
                            ls2 = 1 - ls2

                        img_treated = ls2
###############################################################################################################################################3

                    # Label image regions:
                    label_image, nregions = label(img_treated,return_num=True)
                    ind_regions = np.arange(1,nregions+1)

                    # No regions have been found
                    if nregions == 0:
                        dic_data_table = {'Type': name_coordinates[tt], 'Coordinates': 'Not Found'}
                        lista_rows_segmented.append(dic_data_table)
                        pass
                    # There are different regions
                    else:

                        props = regionprops(label_image)

                        # Area and centroid of each region
                        area = []
                        Centroid_list = []
                        for p in props:
                            area.append(p.area)

                            y0, x0 = p.centroid
                            y0_norm = np.uint(y0)
                            x0_norm = np.uint(x0)
                            Centroid_list.append([x0_norm, y0_norm])

                        # Mask polygon
                        mask_polygon = np.zeros(img.shape)
                        xf, yf = np.uint(x), np.uint(y)
                        mask_polygon[yf,xf] = 1
                        xfilled, yfilled = polygon(xf, yf)
                        mask_polygon[yfilled,xfilled] = 1

                        mask_polygon_cut = mask_polygon[y_min:y_max, x_min:x_max]

                        Centroides_goods = []
                        area_goods =[]

                        good_regions = np.zeros(len(Centroid_list))
                        # Area and coordinates of the regions inside the mask
                        for iv in range(len(Centroid_list)):
                            if (mask_polygon_cut[Centroid_list[iv][1], Centroid_list[iv][0]] == 1):
                                Centroides_goods.append(Centroid_list[iv])
                                area_goods.append(area[iv])
                                good_regions[iv] = 1

                        ind_good = np.where((good_regions == 1))
                        ind_gregions = np.squeeze(np.array(ind_good)+1, axis=0)

                        ind_bad = np.where((good_regions == 0))
                        ind_bregions = np.array(ind_bad)+1

                        # Biggest component (the one with the biggest area)
                        if area_goods != []:
                            index_list_biggest = np.argmax(area_goods)
                            index_comp_biggest = ind_gregions[index_list_biggest]

                            # Delete out and small regions
                            label_image[label_image!=index_comp_biggest] = 0

                        # Compute a mask
                        mask = label_image * mask_polygon_cut

                        # SLIC result
                        slic = segmentation.slic(img_cut, n_segments=10, start_label=1)

                        # maskSLIC result
                        m_slic = segmentation.slic(img_cut, n_segments=10, mask=mask, start_label=1)


                        mask_segmentation = np.zeros(img.shape)
                        mask_segmentation[y_min:y_max, x_min:x_max] = m_slic
                        mask_segmentation = np.uint8(mask_segmentation)
                        mask_binarize = np.zeros(img.shape)
                        mask_binarize[mask_segmentation > 0] = 255

                        contours, _ = cv.findContours(np.uint8(mask_binarize), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                        x_coor = []
                        y_coor = []

                        for u in contours[0]:
                            x_coor.append(u[0][0])
                            y_coor.append(u[0][1])


                        num_elements = np.min([len(x_coor), 500])
                        x_coor = np.array(x_coor); y_coor = np.array(y_coor)
                        idx = np.round(np.linspace(0, len(x_coor) - 1, num_elements)).astype(int)
                        x_coor = x_coor[idx]
                        y_coor = y_coor[idx]

                        for uu in range(len(x_coor)):
                            x_str = str(x_coor[uu])
                            y_str = str(y_coor[uu])

                            if uu == 0:
                                coordenates_segmented = 'M'+x_str+','+y_str+'L'

                            elif uu == len(x_coor)-1:
                                coordenates_segmented = coordenates_segmented + x_str+','+y_str+'Z'

                            else:
                                coordenates_segmented = coordenates_segmented + x_str+','+y_str+'L'

                        annotation_store_segmented[filename]['shapes'][tt]['path'] = coordenates_segmented
                        dic_data_table = {'Type': name_coordinates[tt], 'Coordinates': coordenates_segmented}
                        lista_rows_segmented.append(dic_data_table)

                    lista_rows = lista_rows_segmented

                shapes = [table_row_to_shape(sh) for sh in lista_rows]

                fig.update_layout(
                    shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
                    newshape_line_color=color_dict[name_coordinates[tt]],
                    margin=dict(l=0, r=0, b=0, t=0, pad=4),
                    uirevision=True
                )


            return (" ", fig, lista_rows)

        else:
            img = np.ones((550,868))
            fig = px.imshow(img, binary_string=True)
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0, pad=4),
                dragmode="drawclosedpath",
                uirevision=True
            )

            lista_rows = []
#             dic_data_table = {'column_1': 'Empty', 'column_2': 'Empty'}
            dic_data_table = {'Type': 'Empty', 'Coordinates': 'Empty'}
            lista_rows.append(dic_data_table)

            return (" ", fig, lista_rows)




# set the download url to the contents of the annotations-store (so they can be
# downloaded from the browser's memory)
app.clientside_callback(
    """
function(the_store_data) {
    let s = JSON.stringify(the_store_data);
    let b = new Blob([s],{type: 'text/plain'});
    let url = URL.createObjectURL(b);
    return url;
}
""",
    Output("download", "href"),
    [Input("annotations-store", "data")],
)

# click on download link via button
app.clientside_callback(
    """
function(download_button_n_clicks)
{
    let download_a=document.getElementById("download");
    download_a.click();
    return '';
}
""",
    Output("dummy", "children"),
    [Input("download-button", "n_clicks")],
)


# TODO comment the dbc link
# we use a callback to toggle the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

#===============================================================================
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=DEBUG)
#===============================================================================
