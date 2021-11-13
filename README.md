<a href="https://medium.com/@louisgeisler3/doc2map-travel-your-documents-like-a-walk-on-google-map-1e8b827fdc04">
<img src="https://img.shields.io/badge/Medium_Article-black?style=flat&logo=medium&labelColor=black">
</a>
<a href="https://www.linkedin.com/in/louisgeisler/">
<img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue">
</a>

# Doc2Map

Doc2Map is an algorithm for **topic modeling and visualization**. It can read any type of document files, but not OCR them. It will find topics base on the core idea of [Top2Vec](https://github.com/ddangelov/Top2Vec) and hierarchicaly display them on a map similar to a Google Map:
![Leaflet Map](https://user-images.githubusercontent.com/82355033/140191707-94fc6b1d-e997-45ae-bef8-67cc22cd09d6.gif)
[**Live Demo 1 With Wikipedia Dataset**](https://louisgeisler.github.io/Doc2Map/example/SimpleWikipedia/DocMap.html)

[**Live Demo 2 With 20 News Groups**](https://louisgeisler.github.io/Doc2Map/example/20NewsGroups/DocMap.html)

Or on a scatter plot with a munual zoom level:
![Plotly Map](https://user-images.githubusercontent.com/82355033/140194962-0a3a3611-3e39-4ac9-a3a7-f9d84849cbc7.gif)
[**Live Demo 1 With Wikipedia Dataset**](https://louisgeisler.github.io/Doc2Map/example/SimpleWikipedia/PlotlyDocMap.html)

[**Live Demo 2 With 20 News Groups**](https://louisgeisler.github.io/Doc2Map/example/20NewsGroups/PlotlyDocMap.html)

# Why use Doc2Map?

With Doc2Map, you will be able to create beautiful, intuitive, and interactive visuals to summarise your document corpus in a map, similar to Google Map, with topics, clusters, and documents, instead of the names of countries, states, and cities.

Thanks to Apache Tika –a software able to detect and extract and text from over a thousand different file types– allow Doc2Map to read virtually any kind of file.

**Note:** This is not OCR, can’t extract text from pictures.

# Using Doc2Map

There are two ways to use Doc2Vec:
 - Launching directly the python module
 - Importing the Doc2Map library in your script

## Launching Doc2Map Module

Your first option is to directly launch the module. Once launch, you will have to wait a little for the programm to start, then you will be asked what folder you want to analyse:
![image](https://user-images.githubusercontent.com/82355033/140196515-8bb73e47-821c-4adc-a368-6245748356b8.png)
Select the folder with the document you want to cartography.

For the next step, you will have to be patient. Doc2Map will analyse and convert into plain text your docuemnt, then organise them. Depending of the format, the size and the number of documents, it may take a long time...

When finished, two web pages will be automaticaly launch on your browser to show you different cartographies of you documents.

The examples are loaded from HTML files newly created. You can easily find their localization by looking at the address bar of your browser, you will see something like *file://Your/Path/To/Your/Visuals*

These files can easily be exported to another machine, with little of requirements:
 - If your visualization is based on local files, once exported, these files may no longer be accessible by interacting with the visualisation.
 - However, there will be no problem, if you use a common share hard drive with the people you share the visualisations (like it may often be the case in many firms, under the form of a local network).
For the visualisation DocMap.html, you will have to include the files: DocMapdensity.svg and data.js.

## Importing in a Python Script

If you want to use Doc2Map with python, you have first to install it:
```
pip install Doc2Map
```

Then, you will have to import it:
```python
from Doc2Map import Doc2Map
```

# How Does It Work?

Doc2Map is mainly based on the Top2Vec principle, and rely on Plotly and Leaflet to create beautiful visuals.

If you want to know the complete story and working of Doc2Map, I invite you to read the Medium Article about it: <a href="https://medium.com/@louisgeisler3/doc2map-travel-your-documents-like-a-walk-on-google-map-1e8b827fdc04"><img src="https://img.shields.io/badge/Medium_Article-black?style=flat&logo=medium&labelColor=black"></a>
