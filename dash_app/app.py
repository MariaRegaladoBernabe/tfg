import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash_slicer import VolumeSlicer
from dash import dash_table
import plotly.graph_objs as go

import numpy as np
from nilearn import image
from skimage import draw, filters, exposure, measure
from scipy import ndimage

import plotly.graph_objects as go
import plotly.express as px

from skimage import io
from skimage.color import rgb2gray
import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import skimage as sk
from skimage import data, filters, exposure, io, morphology, img_as_ubyte, segmentation
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
from skimage import io
import os

#===============================================================================
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# server = app.server
# ------------------------------------------------------------------------------------------

DEBUG = True

# Definition of grooves
NUM_ATYPES = 23
DEFAULT_FIG_MODE = "layout"
annotation_colormap = px.colors.qualitative.Light24
annotation_types = [
    "Cingulate Sulcus nothing",
    "Cingulate Sulcus left",
    "Cingulate Sulcus right",
    "Central Sulcus left",
    "Central Sulcus right",
    "Precentral Sulcus left",
    "Precentral Sulcus right",
    "Postcentral Sulcus left",
    "Postcentral Sulcus right",
    "Sylvian Fissure left",
    "Sylvian Fissure right",
    "Superior Temporal Sulcus left",
    "Superior Temporal Sulcus right",
    "Temporo-Occipital Sulcus left",
    "Temporo-Occipital Sulcus right",
    "Parieto-Occipital Sulcus nothing",
    "Parieto-Occipital Sulcus left",
    "Parieto-Occipital Sulcus right",
    "Calcarine Fissure nothing",
    "Calcarine Fissure left",
    "Calcarine Fissure right",
    "Superior Frontal Sulcus left",
    "Superior Frontal Sulcus right"
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


#Function to create a dictionary
def dictionary_function(vector):
    """
    Create a dictionary through using the input vector.

    Args:
        - vector

    Returns:
        - vector_dictionary
    """
    vector_dictionary = {}
    for i in range(len(vector)):
        id = vector[i]
        vector_dictionary[id] = i
    return vector_dictionary

# Function to define the path of each image in the Data Base
def definition_filelist(path, babies_identification, weeks, planes):
    """
    Creates a list with the paths of all the files.

    Args:
        - path: Address where the images are located
        - babies_identification: Baby what is there
        - weeks: Weeks defined
        - plans: The plans that are in the taking of the ultrasound
    Returns:
        - file_list_all: List list made up of the paths to be created
        - filelist_cont: A single list with the paths added in a concatenated way
    """
    file_list_all = []
    filelist_cont = []
    for i in babies_identification:
      file_list_baby = []
      for ii in weeks:
        file_list_week = []
        for iii in planes:
          file_list_week.append(path+i+"/"+i+"_"+ii+"EG_SELECCION ESTANDAR/"+i+"_"+iii+"_M_CUT.jpg")
          filelist_cont.append(path+i+"/"+i+"_"+ii+"EG_SELECCION ESTANDAR/"+i+"_"+iii+"_M_CUT.jpg")

        file_list_baby.append(file_list_week)
      file_list_all.append(file_list_baby)

    return file_list_all, filelist_cont


def identification_babies(dirname):
    """
    The babies in the database are defined

    Args:
        - dirname: The path where the database is located is indicated
    Returns:
        - id_babies: Identification of the babies there are.
    """

    id_babies = []
    babies_dictionary_function = {}
    cont_dictionary = 0

    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for i in subfolders:
      id_baby = i.split('/')[-1]
      id_babies.append(i.split('/')[-1])

    return sorted(id_babies)

path = "./assets/Data_Base_update/"

babies = identification_babies(path)
baby_default = babies[0]

babies_dictionary = dictionary_function(babies)

# Definition of the weeks in the app
weeks = ["24", "25", "26", "27", "28", "29", "30", "31", "32"]
week_default = weeks[0]

weeks_dictionary = dictionary_function(weeks)

# Definition of the planes in the app
planes = ["c1", "c2", "c3", "c4", "c5", "c6", "s1", "s2i", "s2d", "s3i", "s3d", "s4i", "s4d"]
plane_default = planes[0]

planes_dictionary = dictionary_function(planes)

# Definition of two lists, in which the first is a list of paths and the second is a list with
# all concatenated paths

filelist_path_agrup, filelist_path_cont = definition_filelist(path, babies, weeks, planes)

# Reading and plot of the initial image of the app in the layout
image_id_baby_before = -1
image_id_week_before = -1
image_id_plane_before = -1

path_img_ini = filelist_path_agrup[0][0][0]

if (os.path.exists(path_img_ini) == True):
    fig = px.imshow(io.imread(path_img_ini))

else:
    img = np.ones((550,868))
    fig = px.imshow(img, binary_string=True)


fig.update_layout(
    newshape_line_color=color_dict[DEFAULT_ATYPE],
    margin=dict(l=0, r=0, b=0, t=0, pad=4),
    dragmode="drawclosedpath",
    uirevision=True
)



# Deffinition Buttons

button_babies = dcc.Dropdown(
    id='babies_button',
    options=[{'label': i, 'value': i} for i in babies],
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

# Card showing the image selected by the doctor
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

# Card that shows the image with the segmentation made by the algorithm once it is executed
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

# Buttons to select baby, week and plane
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
            dbc.Col(
                run_segmentation_button,
            ),
        ),
    ]
)

# Table showing the manual segmentation made by the user
annotated_data_card_Doctor= dbc.Card(
    [
        dbc.CardHeader(html.H5("Annotated data")),
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

# Define Navbar
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

# Define structure (layout) of the app
app.layout = html.Div(
    [
        nav_bar,
        dbc.Container(
            children = [
                dbc.Row([dbc.Col(Buttons_select_children, md = 6), dbc.Col(Button_segmentation)]),
                dbc.Row([dbc.Col(Image_annotation_card_Doctors, md = 6), dbc.Col(annotated_data_card_Doctor)]),
                dbc.Row([dbc.Col(plot_fig_segmented, md = 6), dbc.Col(table_coordinates_segmentation)]),
            ],
            fluid=True,
        ),
    ],
)


# ------------- Define App Interactivity ---------------------------------------------------

@app.callback(
    # The title of the first card is modified: the baby, the week and the plane
    # according to what has been selected in the dropdowns
    Output(component_id="tittle_card_Doctors", component_property="children"),
    [
        Input(component_id="babies_button", component_property="value"),
        Input(component_id="weeks_button", component_property="value"),
        Input(component_id="planes_button", component_property="value"),
    ]
)
def update_output_div(baby, week, plane):
    return 'Groove selection of the Baby {}, week {} and plane {}'.format(baby, week, plane)


@app.callback(
    # It obtains the data of the dropdown of babies, weeks and plans; in addition to the image path
    # and the image is modified as the data of the manual selection table so that the data is displayed
    # that corresponds
    [Output("annotations-table", "data"), Output("image_files", "data")],
    [
        Input("babies_button", "value"),
        Input("weeks_button", "value"),
        Input("planes_button", "value"),
        Input("Image_baby_week_plane", "relayoutData")
    ],
    [
        State("annotations-table", "data"),
        State("image_files", "data"),
        State("annotations-store", "data"),
        State("annotation-type-dropdown", "value"),
    ],
)
def modify_table_entries(
    id_baby,
    id_week,
    id_plane,
    graph_relayoutData,
    annotations_table_data,
    image_files_data,
    annotations_store_data,
    annotation_type,
):


    global image_id_baby_before, image_id_week_before, image_id_plane_before
    global babies_dictionary, weeks_dictionary, planes_dictionary
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if cbcontext == "Image_baby_week_plane.relayoutData":
        debug_print("graph_relayoutData:", graph_relayoutData)
        debug_print("annotations_table_data before:", annotations_table_data)
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
        if annotations_table_data is None:
            return dash.no_update
        else:
            debug_print("annotations_table_data after:", annotations_table_data)
            return (annotations_table_data, image_files_data)

    image_id_baby = babies_dictionary[id_baby]
    image_id_week = weeks_dictionary[id_week]
    image_id_plane = planes_dictionary[id_plane]

    image_files_data["current_baby"] = image_id_baby
    image_files_data["current_week"] = image_id_week
    image_files_data["current_plane"] = image_id_plane
    #image_files_data["current"] %= len(image_files_data["files"])

    if (image_id_baby != image_id_baby_before or image_id_week != image_id_week_before or image_id_plane != image_id_plane_before):
        image_id_baby_before = image_files_data["current_baby"]
        image_id_week_before = image_files_data["current_week"]
        image_id_plane_before = image_files_data["current_plane"]
        # image changed, update annotations_table_data with new data
        annotations_table_data = []
        filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]

        for sh in annotations_store_data[filename]["shapes"]:
            annotations_table_data.append(shape_to_table_row(sh))
        return (annotations_table_data, image_files_data)
    else:
        return (annotations_table_data, image_files_data)

@app.callback(
    # Every time a new manual segmentation is added, it is added to the corresponding path
    # of the current image (Baby, week and plane buttons) and the table is updated adding the new
    # groove
    [Output("Image_baby_week_plane", "figure"), Output("annotations-store", "data")],
    [Input("annotations-table", "data"), Input("annotation-type-dropdown", "value"),
    Input("babies_button", "value"), Input("weeks_button", "value"), Input("planes_button", "value")],
    [State("image_files", "data"), State("annotations-store", "data")],
)
def send_figure_to_graph(
    annotations_table_data, annotation_type,
    num_baby, num_week, num_plane,
    image_files_data, annotations_store
):
    if annotations_table_data is not None:
        filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]
        # convert table rows to those understood by fig.update_layout
        fig_shapes = [table_row_to_shape(sh) for sh in annotations_table_data]
        # find the shapes that are new
        new_shapes_i = []
        old_shapes_i = []
        for i, sh in enumerate(fig_shapes):
            if not shape_in(annotations_store[filename]["shapes"])(sh):
                new_shapes_i.append(i)
            else:
                old_shapes_i.append(i)
        # add timestamps to the new shapes
        for i in new_shapes_i:
            fig_shapes[i]["timestamp"] = time_passed(annotations_store["starttime"])
        # find the old shapes and look up their timestamps
        for i in old_shapes_i:
            old_shape_i = index_of_shape(
                annotations_store[filename]["shapes"], fig_shapes[i]
            )
            fig_shapes[i]["timestamp"] = annotations_store[filename]["shapes"][old_shape_i]["timestamp"]
        shapes = fig_shapes

        if (os.path.exists(filename) == True):
            fig = px.imshow(io.imread(filename), binary_backend="bmp")
        else:
            img = np.ones((550,868))
            fig = px.imshow(img, binary_string=True)

        fig.update_layout(
            shapes=[shape_data_remove_timestamp(sh) for sh in shapes],
            # reduce space between image and graph edges
            newshape_line_color=color_dict[annotation_type],
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawclosedpath",
            uirevision=True
        )

        annotations_store[filename]["shapes"] = shapes
        return (fig, annotations_store)
    else:
        return dash.no_update


@app.callback(
    # We proceed to receive the coordinates of the table created from the manual segmentations to apply them
    # to each one the algorithm and show the results on the two lower cards. In the first with
    # the image and the resulting segmentation and in the second, the table with the coordinates obtained from the algorithm
    [Output(component_id='run_segmentation_cont', component_property='children'), Output("image_to_segment", "figure"), Output('table_segmentation', 'data')],
    [Input(component_id='run_segmentation_button', component_property='n_clicks'),
     Input("babies_button", "value"), Input("weeks_button", "value"), Input("planes_button", "value"),
     Input("image_to_segment", "relayoutData")],
    [State("image_files", "data"), State("annotations-store", "data"),  State("table_segmentation", "data")],
)

def update_output(n_clicks, id_baby, id_week, id_plane, image_to_seg, image_files_data, annotations_store_data, segmentation_store_data):

    update_figure = False
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if n_clicks is not None and cbcontext == "image_to_segment.relayoutData":

        if "shapes" in image_to_seg.keys():
            # this means all the shapes have been passed to this function via
            # graph_relayoutData, so we store them
            segmentation_store_data = [
                shape_to_table_row(sh) for sh in image_to_seg["shapes"]
            ]
            update_figure = True
        elif re.match("shapes\[[0-9]+\].path", list(image_to_seg.keys())[0]):
            # this means a shape was updated (e.g., by clicking and dragging its
            # vertices), so we just update the specific shape
            segmentation_store_data = annotations_table_shape_resize(
                segmentation_store_data, image_to_seg
            )
            update_figure = True

        action = list(image_to_seg.keys())[0]
        if action == 'dragmode':
            update_figure = False

        if update_figure == True:
            debug_print("annotations_table_data after:", segmentation_store_data)
            shapes = [table_row_to_shape(sh) for sh in segmentation_store_data]

            index = int(((list(image_to_seg.keys())[0]).split('[')[1]).split(']')[0])

            type_line = segmentation_store_data[index]['Type']
            filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]
            img = io.imread(filename, as_gray=True)
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


    if n_clicks is None:

        lista_rows = []
        img = np.ones((550,868))
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(
#             newshape_line_color=color_dict[DEFAULT_ATYPE],
            margin=dict(l=0, r=0, b=0, t=0, pad=4),
            dragmode="drawclosedpath",
            uirevision=True
        )

        return (" ", fig, lista_rows)
    else:
        annotation_store_segmented = annotations_store_data
        filename = image_files_data["files"][image_files_data["current_baby"]][image_files_data["current_week"]][image_files_data["current_plane"]]
        if (os.path.exists(filename) == True):
            img = io.imread(filename, as_gray=True)
            fig = px.imshow(img, binary_string=True)
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=0, pad=4),
                dragmode="drawclosedpath",
                uirevision=True
            )


            coordenadas_grooves_manual = []
            coordinates_segmentation = []
            name_coordinates = []
            for sh in annotations_store_data[filename]["shapes"]:
                coordenadas_grooves_manual.append(shape_to_table_row(sh))

            lista_rows = []
            if coordenadas_grooves_manual == []:
                dic_data_table = {'Type': 'Empty', 'Coordinates': 'Empty'}
                lista_rows.append(dic_data_table)
            else:
                for t in coordenadas_grooves_manual:
                    name = t["Type"]
                    coor = t["Coordinates"][1:-1].split('L')

                    coordinates_segmentation.append(coor)
                    name_coordinates.append(name)
                # The coordinates are stored in two different vectors, one for the x axis and the other for the y
                x_coordinates_each_segmentation = []
                y_coordinates_each_segmentation = []
                shapes = []
                lista_rows_segmented = []

                for i in range(len(coordenadas_grooves_manual)):

                    x_contour = []
                    y_contour = []

                    cadena_array = coordinates_segmentation[i]

                    for iii in range(len(coordinates_segmentation[i])):
                      x_i, y_i = coordinates_segmentation[i][iii].split(",")

                      if (iii==0):
                        x_close = float(x_i)
                        y_close = float(y_i)

                      x_contour.append(float(x_i))
                      y_contour.append(float(y_i))

                      if (iii==(len(coordinates_segmentation[i])-1)):
                        x_contour.append(x_close)
                        y_contour.append(y_close)

                    x_coordinates_each_segmentation.append(np.uint(x_contour))
                    y_coordinates_each_segmentation.append(np.uint(y_contour))

                # Defined algorithm process
                radius = 1
                selem = disk(radius)

                for tt in range(len(x_coordinates_each_segmentation)):
                    x = x_coordinates_each_segmentation[tt]
                    y = y_coordinates_each_segmentation[tt]
                    #take region of the groove
                    x_max, x_min = np.uint(np.max(x)), np.uint(np.min(x))
                    y_max, y_min = np.uint(np.max(y)), np.uint(np.min(y))
                    img_cut = img[y_min:y_max, x_min:x_max]
                    img_cut_preproces = (img_cut-np.min(img_cut))/(np.max(img_cut)-np.min(img_cut))

                    # Threshold Local
                    func = lambda arr: arr.mean()
                    func2 = lambda arr: arr.std()
                    binary_image = (img_cut_preproces > threshold_local(img_cut_preproces, 45, 'generic', param=func))
                    closing_image = closing(binary_image, selem)

                    # Definition of structures
                    # label image regions:
                    label_image, nregions = label(closing_image,return_num=True)
                    ind_regions = np.arange(1,nregions+1)

                    # In case it does not detect structure, it is passed giving a message that it does not exist
                    if nregions == 0:
                        dic_data_table = {'Type': name_coordinates[tt], 'Coordinates': 'Not Found'}
                        lista_rows_segmented.append(dic_data_table)
                        pass

                    else:

                        # Define properties of the regions and take the area of this
                        props = regionprops(label_image)
                        area = []
                        Centroid_list = []
                        for p in props:
                            area.append(p.area)
                            y0, x0 = p.centroid
                            y0_norm = np.uint(y0)
                            x0_norm = np.uint(x0)
                            Centroid_list.append([x0_norm, y0_norm])

                        mask_polygon = np.zeros(img.shape)
                        xf, yf = np.uint(x), np.uint(y)
                        mask_polygon[yf,xf] = 1

                        xfilled, yfilled = polygon(xf, yf)
                        mask_polygon[yfilled,xfilled] = 1

                        mask_polygon_cut = mask_polygon[y_min:y_max, x_min:x_max]

                        Centroides_goods = []
                        area_goods =[]

                        # Acceptable structures are those whose centroid is within manual segmentation
                        good_regions = np.zeros(len(Centroid_list))
                        # Area and coordinates of the regions inside the mask
                        for iv in range(len(Centroid_list)):
                            if (mask_polygon_cut[Centroid_list[iv][1], Centroid_list[iv][0]] == 1):
                                Centroides_goods.append(Centroid_list[iv])
                                area_goods.append(area[iv])
                                good_regions[iv] = 1

                        ind_good = np.where((good_regions == 1))
                        ind_gregions = np.squeeze(np.array(ind_good)+1)

                        ind_bad = np.where((good_regions == 0))
                        ind_bregions = np.array(ind_bad)+1

                        # Biggest component (the one with the biggest area)
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

                        # The segmentation is placed in the area of the image that it touches
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

                        # The number of vertices is limited and it is passed to the same format as that obtained in the first table
                        num_elements = np.min([len(x_coor), 250])
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

if __name__ == "__main__":
     app.run_server(debug=True, dev_tools_props_check=False)
#===============================================================================
#if __name__ == "__main__":
#    app.run_server(host='0.0.0.0', port=8050, debug=DEBUG)
