from requests.api import options
from sklearn.datasets import fetch_20newsgroups
import umap
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import os
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from urllib.request import pathname2url
import sys
import scipy.stats as st
import stopwordsiso as stopwords
import logging
import json
import copy
import requests
import warnings
import unidecode
from PIL import Image, ImageDraw, ImageFont

import easygui as g

from fastcluster import *
#import scipy.cluster.hierarchy as sch


import tika
from tika import parser

headers = {
'X-Tika-PDFextractInlineImages': 'true',
'X-Tika-PDFOcrStrategy': 'ocr_and_text_extraction',
}




class Doc2Map:

    doc2mapHTML = r"""
    <!DOCTYPE html>
    <html>
    <head>
        
        <title>Layers Control Tutorial - Leaflet</title>

        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" integrity="sha512-xodZBNTC5n17Xt2atTPuE1HxjVMSvLVW9ocqUKLsCC5CXdbqCmblAshOMAS6/keqq/sMZMZ19scR4PsZChSR7A==" crossorigin=""/>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js" integrity="sha512-XQoYMqMTK8LvdxXYG3nZ448hOEQiglfqkJs1NOQV44cWnUrBc8PkAOcXy20w0vlaXaVUearIOBhiXZ5V3ynxwA==" crossorigin=""></script>
        <script src="data.js"></script>
        <script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>
        <script src='https://api.mapbox.com/mapbox.js/plugins/leaflet-fullscreen/v1.0.1/Leaflet.fullscreen.min.js'></script>
        <link href='https://api.mapbox.com/mapbox.js/plugins/leaflet-fullscreen/v1.0.1/leaflet.fullscreen.css' rel='stylesheet' />
        <script src="https://unpkg.com/leaflet.markerprecluster@1.0.0/markerprecluster-min.js"></script>
        <style>
            .leaflet-control-zoom-display {
                background-color: #fff;
                border-bottom: 1px solid #ccc;
                width: 26px;
                height: 26px;
                line-height: 26px;
                display: block;
                text-align: center;
                text-decoration: none;
                color: black;
                padding-top: 0.3em;
                font: bold 12px/20px Tahoma, Verdana, sans-serif;
            }
            body {
                padding: 0;
                margin: 0;
            }
            html, body, #map {
                height: 100%;
                width: 100%;
                background: white;
            }
            #map .leaflet-div-icon {
                width:0;
                height:0;
                border: 0;
                padding: 0;
            }

            #map .leaflet-div-icon div {
                display:inline-block;
                padding: 3px;
                border: 1px solid #666;
                border-radius: 3px;
                background:rgba(255, 255, 255, 0.705);
                transform:translate(-50%, -50%);
                text-align: center;
            }
        </style>
        <script>
            L.Control.ZoomDisplay = L.Control.extend({
                options: {
                    position: 'topleft'
                },
        
                onAdd: function (map) {
                    this._map = map;
                    this._container = L.DomUtil.create('div', 'leaflet-control-zoom-display leaflet-bar-part leaflet-bar');
                    this.updateMapZoom(map.getZoom());
                    map.on('zoomend', this.onMapZoomEnd, this);
                    return this._container;
                },
        
                onRemove: function (map) {
                    map.off('zoomend', this.onMapZoomEnd, this);
                },
        
                onMapZoomEnd: function (e) {
                    this.updateMapZoom(this._map.getZoom());
                },
        
                updateMapZoom: function (zoom) {
                    if(typeof(zoom) === "undefined"){zoom = ""}
                    this._container.innerHTML = zoom;
                }
            });
        
        
            L.Map.mergeOptions({
                zoomDisplayControl: true
            });
        
            L.Map.addInitHook(function () {
                if (this.options.zoomDisplayControl) {
                    this.zoomDisplayControl = new L.Control.ZoomDisplay();
                    this.addControl(this.zoomDisplayControl);
                }
            });
        
            L.control.zoomDisplay = function (options) {
                return new L.Control.ZoomDisplay(options);
            };
        </script>
    </head>
    <body>
    <div id='map'></div>
    <script>

        var map = L.map('map', {
            zoom: 0,
            center: [0,0],
            minZoom: -6, //Math.min(...Object.keys(dZoomLevel)),
            maxZoom: 30, //Math.max(...Object.keys(dZoomLevel)),
            crs: L.CRS.Simple,
            //continuousWorld: true,
            //maxBounds: bounds,
            //maxBoundsViscosity: 1,
        });
        map.addControl(new L.Control.Fullscreen());
        map.fitBounds(map_bounds);
        L.imageOverlay('DocMapdensity.svg', image_bounds).addTo(map);
        var zoomOffset = map.getZoom()-1;
        map.options.minZoom = zoomOffset;
        map.options.maxZoom = zoomOffset + max_depth+6;
        map.options.zoom = zoomOffset;

        L.Control.ZoomDisplay.prototype.updateMapZoom = function (zoom) {
            if(typeof(zoom) === "undefined"){zoom = ""}
            this._container.innerHTML = zoom - zoomOffset;
        }
        map.fire("zoomend");
        
        var options = {
            showCoverageOnHover: false,
            maxClusterRadius: 100000,
        };

        var leafs = L.markerClusterGroup({
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: true,
            zoomToBoundsOnClick: true,
            singleMarkerMode: false,
            iconCreateFunction: function(cluster) {
                return L.divIcon({ html: "<div><b>"+ dNode[cluster.nodeID].join("<br>") + '</b><br>(' + cluster.getChildCount() + "children)</div>", iconSize: "auto"});
            },
        });

        for (var info of lMarker) {
            //L.circle(info[1], {radius: 0.001}).addTo(map);
            var leaf = L.leaf(info[1], info[2]).bindPopup('<b>'+info[3]+'</b><br>'+info[4]+'<br>'+
                ((info[5])?'<a href="'+info[5]+'" target="_blank">Open the file</a>':"")
                );
            leafs.addLayer(leaf);
        }
        map.addLayer(leafs);
    </script>
    </body>
    </html>
    """

    #==============================================================================================================#

    dynamic_tree = r"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">

        <title>Tree Example</title>

        <style>
        
        .node {
            cursor: pointer;
        }

        .node circle {
        fill: #fff;
        stroke: steelblue;
        stroke-width: 3px;
        }

        .node text {
        font: 12px sans-serif;
        }

        .link {
        fill: none;
        stroke: #ccc;
        stroke-width: 2px;
        }
        
        </style>

    </head>

    <body>

    <!-- load the d3.js library -->	
    <script src="https://d3js.org/d3.v6.min.js"></script>
    <script src="dynamic_tree.js"></script>


    <script>

    // Set the dimensions and margins of the diagram
    var margin = {top: 20, right: 90, bottom: 30, left: 90},
        width = 960 - margin.left - margin.right,
        height = 500 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    // appends a 'group' element to 'svg'
    // moves the 'group' element to the top left margin
    var svg = d3.select("body").append("svg")
        .attr("width", width + margin.right + margin.left)
        .attr("height", height + margin.top + margin.bottom)
    .append("g")
        .attr("transform", "translate("
            + margin.left + "," + margin.top + ")");

    var i = 0,
        duration = 750,
        root;

    // declares a tree layout and assigns the size
    var treemap = d3.tree().size([height, width]);

    // Assigns parent, children, height, depth
    root = d3.hierarchy(treeData, function(d) { return d.children; });
    root.x0 = height / 2;
    root.y0 = 0;

    // Collapse after the second level
    root.children.forEach(collapse);

    update(root);

    // Collapse the node and all it's children
    function collapse(d) {
    if(d.children) {
        d._children = d.children
        d._children.forEach(collapse)
        d.children = null
    }
    }

    function update(source) {

    // Assigns the x and y position for the nodes
    var treeData = treemap(root);

    // Compute the new tree layout.
    var nodes = treeData.descendants(),
        links = treeData.descendants().slice(1);

    // Normalize for fixed-depth.
    nodes.forEach(function(d){ d.y = d.depth * 180});

    // ****************** Nodes section ***************************

    // Update the nodes...
    var node = svg.selectAll('g.node')
        .data(nodes, function(d) {return d.id || (d.id = ++i); });

    // Enter any new modes at the parent's previous position.
    var nodeEnter = node.enter().append('g')
        .attr('class', 'node')
        .attr("transform", function(d) {
            return "translate(" + source.y0 + "," + source.x0 + ")";
        })
        .on('click', click);

    // Add Circle for the nodes
    nodeEnter.append('circle')
        .attr('class', 'node')
        .attr('r', 1e-6)
        .style("fill", function(d) {
            return d._children ? "lightsteelblue" : "#fff";
        });

    // Add labels for the nodes
    nodeEnter.append('text')
        .attr("dy", ".35em")
        .attr("x", function(d) {
            return d.children || d._children ? -13 : 13;
        })
        .attr("text-anchor", function(d) {
            return d.children || d._children ? "end" : "start";
        })
        .text(function(d) { return d.data.name; });

    // UPDATE
    var nodeUpdate = nodeEnter.merge(node);

    // Transition to the proper position for the node
    nodeUpdate.transition()
        .duration(duration)
        .attr("transform", function(d) { 
            return "translate(" + d.y + "," + d.x + ")";
        });

    // Update the node attributes and style
    nodeUpdate.select('circle.node')
        .attr('r', 10)
        .style("fill", function(d) {
            return d._children ? "lightsteelblue" : "#fff";
        })
        .attr('cursor', 'pointer');


    // Remove any exiting nodes
    var nodeExit = node.exit().transition()
        .duration(duration)
        .attr("transform", function(d) {
            return "translate(" + source.y + "," + source.x + ")";
        })
        .remove();

    // On exit reduce the node circles size to 0
    nodeExit.select('circle')
        .attr('r', 1e-6);

    // On exit reduce the opacity of text labels
    nodeExit.select('text')
        .style('fill-opacity', 1e-6);

    // ****************** links section ***************************

    // Update the links...
    var link = svg.selectAll('path.link')
        .data(links, function(d) { return d.id; });

    // Enter any new links at the parent's previous position.
    var linkEnter = link.enter().insert('path', "g")
        .attr("class", "link")
        .attr('d', function(d){
            var o = {x: source.x0, y: source.y0}
            return diagonal(o, o)
        });

    // UPDATE
    var linkUpdate = linkEnter.merge(link);

    // Transition back to the parent element position
    linkUpdate.transition()
        .duration(duration)
        .attr('d', function(d){ return diagonal(d, d.parent) });

    // Remove any exiting links
    var linkExit = link.exit().transition()
        .duration(duration)
        .attr('d', function(d) {
            var o = {x: source.x, y: source.y}
            return diagonal(o, o)
        })
        .remove();

    // Store the old positions for transition.
    nodes.forEach(function(d){
        d.x0 = d.x;
        d.y0 = d.y;
    });

    // Creates a curved (diagonal) path from parent to the child nodes
    function diagonal(s, d) {

        path = `M ${s.y} ${s.x}
                C ${(s.y + d.y) / 2} ${s.x},
                ${(s.y + d.y) / 2} ${d.x},
                ${d.y} ${d.x}`

        return path
    }

    // Toggle children on click.
    function click(event, d) {
        if (d.children) {
            d._children = d.children;
            d.children = null;
        } else {
            d.children = d._children;
            d._children = null;
        }
        update(d);
    }
    }

    </script>
    </body>
    """
    
    def __init__(self, speed="learn", lLanguage = ["en"], lemmatizing = True, ramification = 20, min_count = 3):

        self.lData = []
        self.model = None
        self.stopword = stopwords.stopwords(lLanguage)
        self.speed = speed
        self.ramification = ramification
        self.tika = False
        self.module_path = os.path.dirname(os.path.realpath(__file__))+"/"
        self.execution_path = os.path.dirname(sys.argv[0])+"/"
        self.lemmatize = self.Lemmatizer(lLanguage, self.module_path).lemmatize
        self.lemmatizing = lemmatizing and not(not(self.lemmatize))
        
        self.non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)
        self.min_count = min_count
        
        sys.setrecursionlimit(1000000)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        if not os.path.exists(self.module_path+"DocMap.html"):
            with open(self.module_path+"DocMap.html", "w") as f:
                f.write(self.doc2mapHTML)
        
        if not os.path.exists(self.module_path+"dynamic_tree.html"):
            with open(self.module_path+"dynamic_tree.html", "w") as f:
                f.write(self.doc2mapHTML)
        
        
    class Lemmatizer:
        
        def __init__(self, lLanguage, module_path):
            
            self.module_path = module_path
            self.dLemma = dict()
            for l in lLanguage:
                self.load_lexique(l)
            
        def lemmatize(self, word):
            
            return self.dLemma.get(word, word), word in self.dLemma
            
        def load_lexique(self, lang):
            
            if not os.path.exists(self.module_path+"lexique/lemmatization-"+lang+".txt"):
                https = "https://raw.githubusercontent.com/michmech/lemmatization-lists/master/lemmatization-"+lang+".txt"
                try:
                    r = requests.get(https)
                except:
                    warnings.warn("Can't download the lemmatization dictionnary from:\n\t"+https+"\n try to download it manually.\nThe lemmatization will be disable.")
                    return False
                
                os.makedirs(self.module_path+"lexique")
                with open(self.module_path+"lexique/lemmatization-"+lang+".txt", 'w', encoding='utf-8-sig') as file:
                    file.write(r.text)
            
            with open(self.module_path+"lexique/lemmatization-"+lang+".txt", 'r', encoding='utf-8-sig') as file:
                for line in file.readlines():
                    l = line.replace("\n","").split("\t")
                    if len(l)==2:
                        self.dLemma[l[1]] = str(l[0])
            
            return True
            

    def __search_folder(self, path):
        
        lFile = []
        for (dirpath, dirnames, filenames) in os.walk(path):
            for file in filenames:
                lFile += [os.path.join(dirpath, file).replace("\\","/")]
        return lFile
    

    def __preprocessed_text(self, text):

        text = text.replace('\n',' ') # Replace break line caracter
        text = text.replace('\x0c',' ') # Delete new page caracter
        text = text.translate(self.non_bmp_map)
        text = simple_preprocess(text)
        if self.lemmatizing:
            text = [self.lemmatize(word)[0] for word in text if not word in self.stopword]
        return text
    
    
    def __readfile(self, file):
        
        text = self.__readfile_tika(file)
        return text
    
        
    def __readfile_tika(self, file):
    
        return parser.from_file(file, headers=headers)["content"]
        

    def add_files(self, path):
        
        if not self.tika:
            tika.initVM()
            self.tika = True

        lFile = self.__search_folder(path)
        print("There is",len(lFile),"found!")
        for file in lFile:
            path = os.path.realpath(file)
            text = self.__readfile(path)
            label = os.path.basename(file)
            url = r"file:"+pathname2url(path)
            target = os.path.dirname(path)
            self.add_text(text, label, url, target)
    

    def add_text(self, text, label, target=None, url=None):
    
        self.lData += [{
            "text": self.__preprocessed_text(text),
            "label": label,
            "URL": url,
            "target": target,
        }]
        

    def __embedding(self):
        
        # validate training inputs
        speed = self.speed
        if speed == "fast-learn":
            hs = 0
            negative = 5
            epochs = 40
        elif speed == "learn":
            hs = 1
            negative = 0
            epochs = 40
        elif speed == "deep-learn":
            hs = 1
            negative = 0
            epochs = 400
        elif speed == "test-learn":
            hs = 0
            negative = 5
            epochs = 1
        else:
            raise ValueError("speed parameter needs to be one of: fast-learn, learn or deep-learn ("+str(speed)+")")

        trainD2V = []
        for info in self.lData:
            trainD2V += [TaggedDocument(info["text"].copy(), [str(info["label"])])]
              
        self.model = Doc2Vec(
            vector_size=300,
            epochs=epochs,
            window=15,
            dm=0,
            sample=1e-5,
            dbow_words=1,
            hs=hs,
            negative=negative,
            min_count=self.min_count,
            workers=os.cpu_count(),
        )
        
        self.model.build_vocab(trainD2V)
        
        self.model.train(trainD2V, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        self.lDoc = [doc for doc in self.model.dv.key_to_index.keys()]
        self.lDocEmbedding = [np.array(self.model.dv[doc]) for doc in self.lDoc]

        self.lWord = [word for word in self.model.wv.key_to_index.keys()]
        self.lWordEmbedding = [np.array(self.model.wv[word]) for word in self.lWord]
        

    def __projection(self):
        
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, metric='cosine', verbose=True)
        reducer.fit(self.lDocEmbedding)
        self.lDocEmbedding2D = reducer.transform(self.lDocEmbedding)
        self.lWordEmbedding2D = reducer.transform(self.lWordEmbedding)
    
    
    def numeric_target(self):
        
        num_tar = {t: i for i, t in enumerate(set([info["target"] for info in self.lData]))}
        for info in self.lData:
            info["target"] = num_tar[info["target"]]


    def build(self):

        self.numeric_target()
        self.__embedding()
        self.__projection()
        self.hierarchical_tree()
        self.simplified_tree()
    

    def scatter(self):
        
        fig = go.Figure(go.Scatter(
            x=self.lDocEmbedding2D[:,0],
            y=self.lDocEmbedding2D[:,1],
            mode='markers',
            customdata=[([data["URL"]], [data["label"]]) for data in self.lData],
            marker={
                "color": [info["target"] for info in self.lData],
                "colorscale": "Viridis",
            },
            hovertemplate=(
                "Label: <b>%{customdata[1]}</b><br>"+
                "URL: %{customdata[0]}"+
                "<extra></extra>")
        ))
        """fig.add_trace(go.Scatter(
            x = self.lWordEmbedding2D[:,0],
            y = self.lWordEmbedding2D[:,1],
            text = self.lWord,
            mode = 'markers+text',
            textfont = {"size": 5},
        ))"""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode="pan",
            margin=dict(l=0,r=0,b=0,t=0),
            #showlegend=False,
            autosize=True,
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig.update_layout(
            template="plotly_white",
            dragmode="pan",
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig['layout'].update(margin=dict(l=0,r=0,b=0,t=0))
        js = r"""var myPlot = document.getElementById('{plot_id}');
        myPlot.on('plotly_click', function(data){
            for (var path of data.points[0].customdata[0]){
                window.open(path, '_blank')
            }
            //alert('Closest point clicked:\n\n'+pts);
        });"""

        filename=self.execution_path+"Doc2Map_Scatter.html"
        
        fig.write_html(
            filename,
            include_plotlyjs='cdn',
            include_mathjax='cdn',
            post_script=js,
            config={
                "autosizable":True,
                'scrollZoom': True
            },
        )
        os.system("start "+os.path.realpath(filename))
        
    
        
    def __2D_density_plot(self, v1, v2, N):
        
        stdX = np.std(v1)
        a,b = np.min(v1)-stdX, np.max(v1)+stdX
        stdY = np.std(v2)
        c,d = np.min(v2)-stdY, np.max(v2)+stdY
        x=np.linspace(a,b,N)
        y=np.linspace(c,d,N)
        X,Y=np.meshgrid(x,y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([v1, v2])
        kernel = st.gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        Z -= np.min(Z)
        
        fig = go.Figure(go.Contour(
            z=Z, 
            x=x,
            y=y,
            colorscale=[[0, "rgba(255,255,255,0)"],
                        [1, "rgba(27,106,175,255)"]],
            showscale = False,
            #reversescale=True,
            opacity=0.8,    
            contours={
                "showlines": False,
            },
        ))
        return fig
    
        
    def hierarchical_tree(self):
        
        #G = self.clusterer.condensed_tree_.to_networkx()
        #G = self.clusterer.single_linkage_tree_.to_networkx()
        
        a = linkage(self.lDocEmbedding2D, method='ward', metric='euclidean', preserve_input=True)
        #a = sch.linkage(self.lDocEmbedding2D, method  = "ward")
        
        b = {i: [int(x) for x in l[:2]] for i, l in enumerate([[]]*len(self.lDocEmbedding2D) + a.tolist())}
        G = nx.convert.from_dict_of_lists(b, create_using=nx.DiGraph)

        dN = {int(node): list(map(int,l)) for node, l in nx.convert.to_dict_of_lists(G).items()}
        G = nx.DiGraph(dN)

        root = [v for v, d in G.in_degree() if d==0][0]
        
        #area_max = np.prod(np.std(self.lDocEmbedding2D, axis=0)*2)
        area_max = np.max(np.amax(self.lDocEmbedding2D, axis=0) - np.amin(self.lDocEmbedding2D, axis=0))**2
        
        def add_property_node(node):
            
            node = int(node)
            lSuccessor = list(G.successors(node))
            lLeaf = []
            for successor in lSuccessor:
                lLeaf += add_property_node(successor)
            
            # Is that node a leaf ?
            if lSuccessor==[]:
                topic = [word for word, score in self.model.wv.most_similar([self.lDocEmbedding[node]], topn=5)]
                G.nodes[node].update({
                    "topic":  topic,
                    "x": self.lDocEmbedding2D[node, 0],
                    "y": self.lDocEmbedding2D[node, 1],
                    "z": 30,
                    "bounds": 0,
                })
                return [node]
            else:
                lEmbedding = np.array([self.lDocEmbedding[i] for i in lLeaf])
                centroid = np.mean(lEmbedding, axis=0)
                topic = [word for word, score in self.model.wv.most_similar([centroid], topn=5)]
                l2D = np.array([self.lDocEmbedding2D[i, :] for i in lLeaf])
                r"""
                <=> area = (area_max)/4^zoom
                <=> 4^zoom = (area_max)/area
                <=> log(4^zoom) = log((area_max)/area)
                <=> zoom*log(4) = log((area_max)/area)
                <=> zoom = log((area_max)/area)/log(4)
                <=> zoom = zoom_step * (log((area_max)/area)/log(4) // zoom_step)
                """
                area = np.max(np.amax(l2D, axis=0) - np.amin(l2D, axis=0))**2
                z = np.log((area_max)/np.prod(area))/np.log(4)
                G.nodes[node].update({
                    "x": np.mean(l2D, axis=0)[0],
                    "y": np.mean(l2D, axis=0)[1],
                    "z": z,
                    "bounds": [np.amin(l2D, axis=0)[::-1].tolist(),
                               np.amax(l2D, axis=0)[::-1].tolist()],
                    "approx bound": np.std(l2D, axis=0)*2.5,
                    "topic":  topic,
                })
                return lLeaf
            
        # Add properties to the graph
        add_property_node(root)
        G.nodes[root]["z"]=-1

        # Set all the leaves at the same Z as their parent node
        for n, degree in G.out_degree():
            if degree==0:
                G.nodes[n]["z"] = G.nodes[list(G.predecessors(n))[0]]["z"]

        self.tree = G
        self.root = root


    def simplified_tree(self):

        G = copy.deepcopy(self.tree)
        root = self.root
        
        # Forcing the tree to have discrete value
        for _, n in list(nx.bfs_edges(G, root)):

            # Can't simplify leaves or root
            if G.in_degree(n)==0 or G.out_degree(n)==0:
                continue
            info = G.nodes[n]
            zn = info["z"]
            predecessor = list(G.predecessors(n))[0]
            lSuccessor = list(G.successors(n))
            # Simplify be deleting that node if ?% of its successors are in the same zoom level ?
            if (G.in_degree(n)!=0 and
                    sum([int(G.nodes[successor]["z"])==int(zn) for successor in lSuccessor])/len(lSuccessor)>=1 and
                    sum([G.out_degree(successor)==0 for successor in lSuccessor])/len(lSuccessor)<0.50):
                for successor in lSuccessor:
                    G.add_edge(predecessor, successor)
                    #if G.out_degree(successor)!=0:
                    #    G.nodes[successor]["z"] = int(np.floor(zn))
                G.remove_node(n)
            else:
                # If it is on the same zoom level ?
                if int(zn)==G.nodes[predecessor]["z"]: # or G.out_degree(n)==1:
                    for successor in lSuccessor:
                        G.add_edge(predecessor, successor)
                    G.remove_node(n)
                else:
                    info["z"] = G.nodes[predecessor]["z"] + 1


        while True:
            
            nDiv = 0  
            div = 0
            lDiv = []
            dZoomLevelDensity = dict()
            sZ=set()
            for n, info in list(G.nodes.items()):
                if G.out_degree(n)!=0:
                    lS = [G.out_degree(s)==0 for s in G.successors(n)]
                    if all(lS):
                        div += len(lS)
                        nDiv += 1
                        lDiv += [len(lS)]
                        sZ.add(int(info["z"]))
                else:
                    dZoomLevelDensity[int(info["z"])] = dZoomLevelDensity.get(int(zn), 0) + 1


            # Stop when the mean of leaves by parents is greater than 12
            if div/nDiv>self.ramification:
                break
            
            # Cut all the leaves after a maximal depth
            #max_zoom = max(list(dZoomLevelDensity.items()), key=lambda x: x[1])[0]-1
            max_zoom = max(sZ)
            
            for n, info in list(G.nodes.items()):
                if info["z"]>=max_zoom and G.out_degree(n)!=0 and G.in_degree(n)!=0:
                    p = list(G.predecessors(n))[0]
                    for s in G.successors(n):
                        G.add_edge(p, s)
                        #if G.out_degree(s)!=0:
                        #    G.nodes[s]["z"] = int(np.floor(zn))
                    G.remove_node(n)

        # Set all the leaves at the same Z as their parent node
        for n, degree in G.out_degree():
            if degree==0:
                G.nodes[n]["z"] = G.nodes[list(G.predecessors(n))[0]]["z"]

        self.simplified_tree = G
        
        
    def interactive_tree(self, G=None, root=None):
        
        if not G: G=self.simplified_tree
        if not root: root=self.root
        
        def recursive(node):
            
            # Is it a leaf?
            if G.out_degree(node)==0:
                return {
                    "name": " ".join(G.nodes[node]["topic"]),
                }
            else:
                return {
                    "name": " ".join(G.nodes[node]["topic"]),
                    "children": [
                        recursive(child)
                        for child in G.successors(node)
                    ]
                }
        
        with open(self.execution_path+"dynamic_tree.json", "w") as f:
            f.write("var treeData = ["+str(recursive(self.root))+"];")
            #json.dump(f, recursive(self.root))
        
        with open(self.execution_path+"dynamic_tree.html", "w") as f:
            f.write(self.dynamic_tree)
        
        os.system("start "+os.path.realpath(self.execution_path+"dynamic_tree.html"))
        
        
        
        
    
    def plotly_interactive_map(self, G=None, root=None):
        
        def cluster(node, lLeaf, image=True):
            
            
            fig = go.Figure(go.Scatter(
                y = [self.lDocEmbedding2D[i,0] for i in lLeaf],
                x = [self.lDocEmbedding2D[i,1] for i in lLeaf],
                mode = 'markers',
                #marker = {"size": 0.7}
                customdata=[([data["URL"]], [data["label"]]) for data in self.lData],
                hovertemplate=(
                    "Label: <b>%{customdata[1]}</b><br>"+
                    "URL: %{customdata[0]}"+
                    "<extra></extra>")
            ))
            
            topic = G.nodes[node]["topic"]
            L, H = G.nodes[node]["approx bound"]
            centroid = (G.nodes[node]["y"], G.nodes[node]["x"])
            
            arial = ImageFont.truetype("arial.ttf", 40)
            
            lMot = [unidecode.unidecode(mot) for mot in topic]
            
            lMot2=[]
            f = max(1, round(L/H))
            lMax = 0
            hTot = 0
            for i in range(0, len(lMot), f):
                ligne=" ".join(lMot[:f])
                lMot2+=[ligne]
                lMot = lMot[f:]
                l, h = arial.getsize(ligne)
                lMax = max(l, lMax)
                hTot += h
            lMot=lMot2
            
            text="\n".join(lMot)
            
            spacing = 0
            
            l, h = lMax, hTot+spacing*(len(topic)-1)

            im = Image.new("RGBA", (l, h), (255,255,255,0))
            d = ImageDraw.Draw(im)
            d.multiline_text((0, 0), text, spacing=spacing, fill="black", font=arial, align='center')
            
            fig.add_layout_image(
                dict(
                    source=im,
                    xref="x",
                    yref="y",
                    x=centroid[0],
                    y=centroid[1],
                    sizex=L,
                    sizey=H,
                    sizing="contain", #['fill', 'contain', 'stretch']
                    opacity=1,
                    layer="above",
                    xanchor="center",
                    yanchor="middle",
                )
            )
            
            return fig
        
        if not G: G=self.simplified_tree
        if not root: root=self.root
        
        
        dFrame = dict()
        
        def build_plot(node):
            
            node = int(node)
            lSuccessor = list(G.successors(node))
            lLeaf = []
            for successor in lSuccessor:
                lLeaf += build_plot(successor)
            
            # Is that node a leaf ?
            if lSuccessor==[]:
                return [node]
            else:
                dFrame[G.nodes[node]["z"]] = dFrame.get(G.nodes[node]["z"], []) + [cluster(node, lLeaf)]
                return lLeaf
        
        build_plot(root)
        
        fig_dict = {
            "data": [],
            "layout": {
                "images": []
            },
        }

        lVisibleData = []
        lVisibleLayout = []
        lName = []        
                
        for z in sorted(list(dFrame.keys())):
            
            frame = dFrame[z]
            lName += [str(z)]
            
            a = len(fig_dict["data"])
            b = len(frame)
            lVisibleData+=[list(range(a, a + b))]
            
            lVisibleLayout += [[]]
            for i, f in enumerate(frame):
                lVisibleLayout[-1] += list(f["layout"]["images"])
                fig_dict["data"] += f["data"]
        
        
        for i in range(len(fig_dict["data"])):
            fig_dict["data"][i]["visible"] = False
        
        for i in lVisibleData[0]:
            fig_dict["data"][i]["visible"] = True
        
        # Create and add slider
        steps = []
        for i, (nom, lD, lL) in enumerate(zip(lName, lVisibleData, lVisibleLayout)):
            step = dict(
                label=nom,
                method="update",
                args=([{"visible": [False]*len(fig_dict["data"])},
                    {"images": lL,
                    "title": "Number of Topics: " + str(len(lL)),
                    "showlegend": False,
                    }]
                    if lL else
                    [{"visible": [False]*len(fig_dict["data"])},
                    {"title": "Série",
                    "images": [],
                    "showlegend": True,
                    }]),
            )
            for j in lD:
                step["args"][0]["visible"][j] = True
            
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Zoom Level: "},
            pad={"t": 50},
            steps=steps
        )]
        
        fig_dict["layout"]["images"] = lVisibleLayout[0]
        fig = go.Figure(fig_dict)

        fig.update_layout(
            sliders=sliders
        )

        fig.update_layout(
            template="plotly_white",
            dragmode="pan",
            xaxis= {
                'showgrid': False, # thin lines in the background
                'zeroline': False, # thick line at x=0
                'visible': False,  # numbers below
            },
            yaxis= {
                'showgrid': False, # thin lines in the background
                'zeroline': False, # thick line at x=0
                'visible': False,  # numbers below
            },
        )

        fig.update(
            layout_showlegend=False
        )

        js = r"""var myPlot = document.getElementById('{plot_id}');
        myPlot.on('plotly_click', function(data){
            for (var path of data.points[0].customdata[0]){
                window.open(path, '_blank')
            }
            //alert('Closest point clicked:\n\n'+pts);
        });"""

        filename = self.execution_path + "PlotlyDocMap.html"
        fig.write_html(
            filename,
            post_script=js,
            include_plotlyjs='cdn',
            include_mathjax='cdn',
            config={
                "autosizable":True,
                'scrollZoom': True
            },
        )
        
        os.system("start "+os.path.realpath(filename))
                
        
    
    def interactive_map(self, G=None, root=None):
        
        if not G: G=self.simplified_tree
        if not root: root=self.root

        map_background = self.execution_path+"DocMapdensity.svg"
        
        fig = self.__2D_density_plot(self.lDocEmbedding2D[:,1], self.lDocEmbedding2D[:,0], 200)
        fig.add_trace(go.Scatter(
            y = self.lDocEmbedding2D[:,0],
            x = self.lDocEmbedding2D[:,1],
            mode = 'markers',
            marker = {"size": 0.7,
                    "color": [info["target"] for info in self.lData],
                    "colorscale": "Viridis",
                    },
        ))
        """fig.add_trace(go.Scatter(
            y = self.lWordEmbedding2D[:,0],
            x = self.lWordEmbedding2D[:,1],
            text = self.lWord,
            mode = 'markers+text',
            textfont = {"size": 5},
        ))"""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode="pan",
            margin=dict(l=0,r=0,b=0,t=0),
            showlegend=False,
            autosize=True,
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1,
            showgrid=False,
            zeroline=False,
            visible=False,
        )
        fig.write_image(map_background)
        
        full_fig = fig.full_figure_for_development(warn=False)
        xbound = full_fig.layout.xaxis.range
        ybound = full_fig.layout.yaxis.range
        image_bounds = [[ybound[1], xbound[0]], [ybound[0], xbound[1]]]

        max_depth = 0
        lMarker = []
        
        for node, info in G.nodes.items():
            if G.out_degree(node)==0:
                path = nx.shortest_path(G, root, node)
                max_depth = max(max_depth, len(path))
                lMarker += [[node, [info["x"], info["y"]], path, self.lData[node]["label"], info["topic"], self.lData[node]["URL"]]]
        
        dNode = {node: info["topic"] for node, info in G.nodes.items()}
        
        map_bounds = [np.amin(self.lDocEmbedding2D, axis=0).tolist(),
                  np.amax(self.lDocEmbedding2D, axis=0).tolist()]
        
        with open(self.execution_path+"data.js", 'w', encoding="UTF-8") as file:
            file.write(("const root="+str(root)+";"
                        +"\nconst image_bounds="+str(image_bounds)+";"
                        +"\nconst map_bounds="+str(map_bounds)+";"
                        +"\nconst lMarker="+str(lMarker).replace("None", "null")+";"
                        +"\nconst dNode="+str(dNode)+";"
                        +"\nconst max_depth="+str(max_depth)+";"
                        ))
            
        with open(self.module_path+"DocMap.html", 'r') as file:
            html = file.read()
        
        with open(self.execution_path+"DocMap.html", 'w') as file:
            file.write(html)
        
        os.system("start "+os.path.realpath(self.execution_path+"DocMap.html"))
    
    
    def display_tree(self):
        self.display_graph(self.tree, self.root, "tree.html")
    
    
    def display_simplified_tree(self):
        self.display_graph(self.simplified_tree, self.root, "simplified_tree.html")
        
    
    def display_graph(self, G, root, filename="tree.html"):
        
        dLigne = {"x": [], "y": [], "z":[]}
        
        for a, b in G.edges.keys():
        
            a = G.nodes[a]
            b = G.nodes[b]
            dLigne["x"] += [a["x"], b["x"], None]
            dLigne["y"] += [a["y"], b["y"], None]
            dLigne["z"] += [a["z"], a["z"], None]
            
            dLigne["x"] += [b["x"], b["x"], None]
            dLigne["y"] += [b["y"], b["y"], None]
            dLigne["z"] += [a["z"], b["z"], None]
            
        
        dNode = {"x": [], "y": [], "z":[], "topic": []}
        dLeaf = {"x": [], "y": [], "z":[], "topic": []}
        
        zoom_min = G.nodes[root]["z"]
        zoom_max = zoom_min
        
        for n, node in G.nodes.items():
            
            # Is leaf ?
            if G.out_degree(n)==0:
                dLeaf["x"] += [node["x"]]
                dLeaf["y"] += [node["y"]]
                dLeaf["z"] += [node["z"]]
                dLeaf["topic"] += [" ".join(node["topic"])]
            else:
                dNode["x"] += [node["x"]]
                dNode["y"] += [node["y"]]
                dNode["z"] += [node["z"]]
                dNode["topic"] += [" ".join(node["topic"])]
                
            zoom_max = max(zoom_max, node["z"])
            zoom_min = min(zoom_min, node["z"])
        
        lZoom_level = np.arange(zoom_min, zoom_max, 1)
        
        #fig = self.__2D_density_plot(self.lDocEmbedding2D[:,1], self.lDocEmbedding2D[:,0], 400)
        fig = go.Figure(go.Scatter3d(
            x=dLigne["x"],
            y=dLigne["y"],
            z=dLigne["z"],
            mode='lines',
            line=dict(
                color='rgba(125,125,125,100)',
                width=0.5,
            ),
            hoverinfo='none',
        ))
        fig.add_trace(go.Scatter3d(
            x=dNode["x"],
            y=dNode["y"],
            z=dNode["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=6,
                color = "red",
            ),
            customdata=[([data]) for data in dNode["topic"]],
            hovertemplate=(
                "Topic: <b>%{customdata[0]}</b><br>"+
                "Z: %{z:0.7f}"+
                "<extra></extra>")
        ))
        fig.add_trace(go.Scatter3d(
            x=dLeaf["x"],
            y=dLeaf["y"],
            z=dLeaf["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=6,
                color = "blue",
            ),
            customdata=[([data["URL"]], [data["label"]]) for data in self.lData],
            hovertemplate=(
                "Label: <b>%{customdata[1]}</b><br>"+
                "URL: %{customdata[0]}"+
                "<extra></extra>")
        ))
        fig.update_layout(
            scene = dict(
                aspectratio=dict(x=2, y=2, z=1),
                xaxis=dict(
                    title='',
                    showticklabels=False,
                    spikecolor='#1fe5bd',
                ),
                yaxis=dict(
                    title='',
                    showticklabels=False,
                    spikecolor='#1fe5bd',
                ),
                zaxis = dict(
                    #tickmode = 'array',
                    ticktext=[str(int(z)) for z in lZoom_level],
                    tickvals=lZoom_level,
                    ticks='outside',
                    tickwidth=4,
                    title='Zoom Level',
                    spikecolor='#1fe5bd',
                ),
            ),
        )
        fig.update_layout(
            margin=dict(l=0,r=0,b=0,t=0),
            showlegend=False,
            autosize=True,
        )

        fig.update_scenes(camera_projection_type='orthographic')
        
        path = self.execution_path+filename

        js = r"""var myPlot = document.getElementById('{plot_id}');
        myPlot.on('plotly_click', function(data){
            for (var customdata of data.points[0].customdata){
                if (customdata.constructor === Array) {
                    window.open(customdata[0], '_blank');
                }
            }
            //alert('Closest point clicked:\n\n'+pts);
        });"""
        
        fig.write_html(
            path,
            include_plotlyjs='cdn',
            include_mathjax='cdn',
            post_script = js,
            config={
                "autosizable":True,
                'scrollZoom': True
            },
        )
        
        os.system("start "+os.path.realpath(path))


    @classmethod
    def test_20newsgroups(cls, speed="learn", subset='test', ramification=22):
        
        d2m = cls(speed=speed, ramification=ramification)
        dataset = fetch_20newsgroups(subset=subset, shuffle=True, random_state=42)

        for i, (data, target) in enumerate(zip(dataset.data, dataset.target)):
            d2m.add_text(data, str(i), target=target)
        
        d2m.build()
        d2m.display_tree()
        d2m.display_simplified_tree()
        d2m.scatter()
        d2m.plotly_interactive_map()
        d2m.interactive_map()
    
        
    @classmethod
    def test_simplewiki(cls):
        
        if not os.path.exists(os.path.dirname(sys.argv[0])+"/simplewiki.json"):
            print("The simplewiki.json isn't here. Please download it from that web page: https://www.kaggle.com/louisgeisler/simple-wiki?select=simplewiki.json")
            os.system(r"""start https://www.kaggle.com/louisgeisler/simple-wiki?select=simplewiki.json""")
            return
            
        with open("simplewiki.json", "r", encoding='utf-8') as f:
            lData = json.load(f)
        
        p1 = np.percentile([len(data["content"]) for data in lData], 60)
        p2 = np.percentile([len(data["content"]) for data in lData], 90)
        lData = [data for data in lData if p1<len(data["content"])<p2]
        lData = np.random.choice(lData, 1000, replace=False)
        n = len(lData)
        percent = 5*n/100
        d2m = cls()
        for i, info in enumerate(lData):
            if (i%percent==0):
                print((100*i)//percent)
            d2m.add_text(info["content"], info["title"], url = info["url"].replace("http://s.wikipedia.org/","https://simple.wikipedia.org/"))
               
        d2m.build()
        d2m.display_tree()
        d2m.display_simplified_tree()
        d2m.scatter()
        d2m.plotly_interactive_map()
        d2m.interactive_map()
        
    
    @classmethod
    def main(cls):
    
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        
        path = g.diropenbox("Folder of the documents to analyse:", "Folder of the documents to analyse:")
        
        d2m = Doc2Map()
        d2m.add_files(path)
        d2m.build()
        d2m.plotly_interactive_map()
        d2m.interactive_map()


if __name__ == "__main__":
    
    Doc2Map.main()