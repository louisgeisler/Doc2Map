L.DistanceGrid = function (zoom, minZoom, maxZoom) {
	this._zoom = zoom;
	this._level = zoom - minZoom;
	this._dict = {};
};

L.DistanceGrid.prototype = {

	addObject: function (marker, point) {
		var group = this._getGroup(marker);
		if (group!=null) {
			this._dict[group] = marker;
		}
	},
	
	_getGroupRecursive: function (marker) {
		if (marker.hierarchy && (marker.hierarchy.length >= this._level)) {
			return marker.hierarchy[this._level]
		} else if (marker._childClusters) {
			var children = marker._markers.concat(marker._childClusters);
			if (children.length) {
				return this._getGroupRecursive(children[0]);
			}
		}
		return null;
	},

	_getGroup: function (marker) {
		var nodeID = this._getGroupRecursive(marker);
		marker.nodeID = nodeID;
		return nodeID;
	},

	//Returns true if the object was found
	removeObject: function (marker, point) {
		var group = this._getGroup(marker);
		if (group && group in this._dict) {
			delete this._dict[group];
			return true;
		} else {
			return false;
		}
	},

	getNearObject: function (marker) {
		var group = this._getGroup(marker);
		if (group in this._dict) {
			return this._dict[group];
		} else {
			//No marker found
			return null;
		}
	},
};

L.leaf = function (position, hierarchy, ...options) {
	var obj = L.marker(position, ...options);
	obj.hierarchy = hierarchy;
	return obj;
};

L.MarkerClusterGroup.prototype._generateInitialClusters = function () {
	var maxZoom = Math.ceil(this._map.getMaxZoom()),
		minZoom = Math.floor(this._map.getMinZoom()),
		radius = this.options.maxClusterRadius,
		radiusFn = radius;

	//If we just set maxClusterRadius to a single number, we need to create
	//a simple function to return that number. Otherwise, we just have to
	//use the function we've passed in.
	if (typeof radius !== "function") {
		radiusFn = function () { return radius; };
	}

	if (this.options.disableClusteringAtZoom !== null) {
		maxZoom = this.options.disableClusteringAtZoom - 1;
	}
	this._maxZoom = maxZoom;
	this._gridClusters = {};
	this._gridUnclustered = {};

	//Set up DistanceGrids for each zoom
	for (var zoom = maxZoom; zoom >= minZoom; zoom--) {
		this._gridClusters[zoom] = new L.DistanceGrid(zoom, minZoom, maxZoom);
		this._gridUnclustered[zoom] = new L.DistanceGrid(zoom, minZoom, maxZoom);
	}

	// Instantiate the appropriate L.MarkerCluster class (animated or not).
	this._topClusterLevel = new this._markerCluster(this, minZoom - 1);
};

L.MarkerClusterGroup.prototype._addLayer = function (layer, zoom) {
	var gridClusters = this._gridClusters,
		gridUnclustered = this._gridUnclustered,
		minZoom = Math.floor(this._map.getMinZoom()),
		markerPoint, z;

	if (this.options.singleMarkerMode) {
		this._overrideMarkerIcon(layer);
	}

	layer.on(this._childMarkerEventHandlers, this);

	//Find the lowest zoom level to slot this one in
	for (; zoom >= minZoom; zoom--) {
		markerPoint = this._map.project(layer.getLatLng(), zoom); // calculate pixel position

		//Try find a cluster close by
		var closest = gridClusters[zoom].getNearObject(layer);
		if (closest) {
			closest._addChild(layer);
			layer.__parent = closest;
			return;
		}

		//Try find a marker close by to form a new cluster with
		closest = gridUnclustered[zoom].getNearObject(layer);
		if (closest) {
			var parent = closest.__parent;
			if (parent) {
				this._removeLayer(closest, false);
			}

			//Create new cluster with these 2 in it
			var newCluster = new this._markerCluster(this, zoom, closest, layer);
			gridClusters[zoom].addObject(newCluster, this._map.project(newCluster._cLatLng, zoom));
			closest.__parent = newCluster;
			layer.__parent = newCluster;
			
			//First create any new intermediate parent clusters that don't exist
			var lastParent = newCluster;
			for (z = zoom - 1; z > parent._zoom; z--) {
				lastParent = new this._markerCluster(this, z, lastParent);
				gridClusters[z].addObject(lastParent, this._map.project(closest.getLatLng(), z));
			}
			parent._addChild(lastParent);

			//Remove closest from this zoom level and any above that it is in, replace with newCluster
			this._removeFromGridUnclustered(closest, zoom);

			return;
		}

		//Didn't manage to cluster in at this zoom, record us as a marker here and continue upwards
		gridUnclustered[zoom].addObject(layer, markerPoint);
	}

	//Didn't get in anything, add us to the top
	this._topClusterLevel._addChild(layer);
	layer.__parent = this._topClusterLevel;
	return;
};
