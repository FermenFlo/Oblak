import os
import numpy as np
from partition import Partition
from keras.models import load_model

class Model:
    MODEL_FOLDER = '../serialization/models/'
    
    def __init__(self, model_file_name):
        self.model_path = os.path.join(self.MODEL_FOLDER, model_file_name)
        self.model = load_model(self.model_path, compile=False)
        self.input_shape = self._infer_input_shape()
        self.max_partition_size = self._infer_max_input_shape()  # Assume all models receive square input and are 2D.
        self.output_type = Partition if self.model.layers[0].output_shape == self.model.layers[-1].output_shape else float
        
    
    def _infer_input_shape(self):
        if len(self.model.input_shape) <= 2:
            return (self.model.input_shape[-1],)
        
        else:
            return tuple([x if x else -1 for x in self.model.input_shape])
        
    def _infer_max_input_shape(self):
        if len(self.model.input_shape) <= 2:
            return int(self.model.input_shape[-1] ** (1/2))
        
        else:
            return self.input_shape[-2]
    
    def _partition_to_input(self, partition):
        """ Translates partitions into a format that is acceptable for the particular model. """
        
        return partition.fit_matrix(self.max_partition_size).reshape(self.input_shape)
    
    def _output_to_partition(self, output):
        parts = []
        for row in output:
            row_sum = sum(row)
            if not row_sum:
                break
            parts.append(int(row_sum))
        
        return Partition(parts)
        
    def _output_to_float(self, output, prob = False):
        """ Translates model output into a single value as expected. """
        actual_output = output[0][0]
        
        return actual_output if prob else actual_output.round()
    
    def predict(self, partition, cast = None):
        model_input = np.array([self._partition_to_input(partition)])
#         model_output = self._output_to_float(self.model.predict_proba(model_input))
        model_output = self._output_to_partition(self.model.predict(model_input))
    
        return cast(model_output) if cast else model_output