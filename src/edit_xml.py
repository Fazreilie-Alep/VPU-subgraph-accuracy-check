#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from lxml import etree as et, objectify
from pathlib import Path
from shutil import copyfile


def changeEdges(edges, legimate_paths, layerid, resultid):
    # find the first edge with id

    layer_edge = edges.find('''edge[@from-layer='{0}']'''.format(layerid))

    if layer_edge is None:
        raise Exception("Cannot find edge for layer")

    layer_edge.set("to-layer", resultid)
    layer_edge.set("to-port", '0')

    # del unused graph
    for sibling in layer_edge.itersiblings():
        edges.remove(sibling)

    # go up to the graph and delete interrupted paths
    paths = set()

    legimate_paths.add(layerid)
    legimate_paths.add(layer_edge.get("from-layer"))
    for sibling in layer_edge.itersiblings(preceding=True):
        # get ids
        current_id = sibling.get("to-layer")
        parent_id = sibling.get("from-layer")

        # if id is in path - add parent
        if current_id in legimate_paths:
            legimate_paths.add(parent_id)

            if parent_id in paths:
                paths.remove(parent_id)
            continue

        # cases for delete
        if current_id < layerid:
            paths.add(current_id)

        if parent_id not in legimate_paths and parent_id < layerid:
            paths.add(parent_id)
        edges.remove(sibling)


def delLayers(tree, layerName):

    # get layer by name
    layer = tree.find('''//layer[@name='{0}']'''.format(layerName))

    if layer is None:
        raise Exception('''Cannot find layer by name {0}'''.format(layerName))

    # there's nothing to cut
    if layer.get("type") == "Result":
        return

    output = layer.find("output")
    # there must be at least 1 output
    if not output:
        raise Exception('''Cannot find output for layer: {0}'''.format(layerName))

    # TODO let's assume we have only one output port (legacy)
    output_port = output.find("port")
    if not output_port:
        raise Exception('''Cannot find output port for layer: {0}'''.format(layerName))

    # remember port dimensions: copies of XML nodes are required
    # otherwise nodes modification will affect original nodes
    output_port_dims = [et.fromstring(et.tostring(dim)) for dim in output_port.iter("dim")]

    # get candidate result layer
    result = tree.find("//layer[@type='Result']")

    if result is None:
        raise Exception("Cannot find result layer")

    # change graph
    edges = tree.find("edges")
    if edges is None:
        raise Exception("Cannot find edges in IR")

    # paths = set()
    legimate_path = set()

    # change graph and search for unused paths
    changeEdges(edges, legimate_path, layer.get("id"), result.get("id"))

    legimate_path.add(result.get("id"))

    result_input = result.find("input")
    if not result_input:
        raise Exception('''Cannot find input for Result id: {0}'''.format(result.get("id")))

    result_input_port = result_input.find("port")
    if not result_input_port:
        raise Exception('''Cannot find input port for Result id: {0} '''.format(result.get("id")))

    # override dims for new Result by actuals Layer dims:
    # As first step we remove current dims in Result
    # and then restore dims from Layer as new dims for Result
    for dim in result_input_port.iter("dim"):
        result_input_port.remove(dim)
    for dim in output_port_dims:
        result_input_port.append(dim)

    print("OpenVINO fix: set attribute `names` forcibly for output with `port_id`: {0} in `layer`: {1}".format(
        output_port.get("id"), layer.get("name")))
    output_port.set("names", result.get("name"))

    path_postfix = '''not(contains(concat('|', '{0}', '|'), concat('|', @id, '|')))'''.format('|'.join(legimate_path)) if len(
        legimate_path) > 1 else '''(@id < '{0}' and @id > '{1}')'''.format(result.get("id"), layer.get("id"))
    layer_xpath = '''layer[{0}]'''.format(path_postfix)

    # del unnessecary layers
    parent = layer.getparent()

    for ll in parent.xpath(layer_xpath):
        parent.remove(ll)
