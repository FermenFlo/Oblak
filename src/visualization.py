from src.models import Model
from src.partition import Partition
from src.generators import PartitionClass, OblakClass

import plotly as py
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self, model_path, full_path = False):
        self.model = Model(model_path, full_path)
        self.encoder = self.model.model.layers[1]
        
        
    def _partitions_to_inputs(self, partitions):
        return np.array([
            partition.fit_matrix(self.model.max_partition_size).reshape(self.model.input_shape[-3:-1])
            for partition in partitions
        ]).reshape(self.model.input_shape)
        
    def visualize_multiple(self, partition_lists, dimension = 3, lines = False, color = None, fig = None):
        
        fig = fig if fig else go.Figure()
        
        for partition_list in partition_lists:
            fig = self.visualize(
                partitions = partition_list,            
                dimension = dimension,         
                lines = lines,
                color = color,            
                fig = fig
            )
        
        return fig
    
    def visualize(self, partitions, dimension = 3, lines = False, color = None, fig = None):
        partition_inputs = self._partitions_to_inputs(partitions)
        colorings = [color(p) for p in partitions] if color else []
       
        fig = fig if fig else go.Figure()
    
        if dimension == 3:
            z_mean, _, _ = self.encoder.predict(partition_inputs)
            
            fig.add_trace(go.Scatter3d(
                x=z_mean[:, 0],
                y = z_mean[:, 1],
                z = z_mean[:, 2],
                name="z",
                mode= None if lines else 'markers',
                text = [repr(p) for p in partitions],
                marker=dict(
                    size=10,
                    color = colorings,
                    colorscale='Viridis',
                )
            ))

            fig.update_layout(
                autosize=False,
                width=1000,
                height=1000,
                scene = dict(
                    xaxis = dict(nticks=4, range=[-4, 4],),
                    yaxis = dict(nticks=4, range=[-4, 4],),
                    zaxis = dict(nticks=4, range=[-4, 4],),
                )
            )
            
        return fig