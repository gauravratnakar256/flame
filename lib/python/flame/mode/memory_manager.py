from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker
from collections import OrderedDict
import torch
import numpy as np

def remove_shm_from_resource_tracker():

        def fix_register(name, rtype):
            if rtype == "shared_memory":
                return
            return resource_tracker._resource_tracker.register(self, name, rtype)
        resource_tracker.register = fix_register

        def fix_unregister(name, rtype):
            if rtype == "shared_memory":
                return
            return resource_tracker._resource_tracker.unregister(self, name, rtype)
        resource_tracker.unregister = fix_unregister

        if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
            del resource_tracker._CLEANUP_FUNCS["shared_memory"]

class MemoryManager():

    def __init__(self, task_id):
        remove_shm_from_resource_tracker()
        self.shm_dict = {}
        self.model_structure = OrderedDict()
        self.task_id = task_id

    def __create_structure(self, parameters, layer_name):
        for name, param in parameters:
             numpy_array = torch.clone(param).detach().numpy()
             numpy_array_datatype = numpy_array.dtype
             mem_size = int(numpy_array.nbytes)
             parameter_name =  layer_name + "." + name
             shared_mem_name = self.task_id + "." + layer_name + "." + name
             shm = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=mem_size)
             self.shm_dict[shared_mem_name] = shm
             self.model_structure[parameter_name] = {'memsize': mem_size, 'dtype': numpy_array_datatype,'shape': numpy_array.shape}


    def create_model_structure(self, model):
        for layer_name, module in model.named_modules():
            self.__create_structure(module.named_parameters(recurse=False), layer_name)
            self.__create_structure(module.named_buffers(recurse=False), layer_name)

    def load_parameters_to_shared_memory(self, model):
        for layer_name, module in model.named_modules():
            self.__load_parameters(module.named_parameters(recurse=False), layer_name, module, True)
            self.__load_parameters(module.named_buffers(recurse=False), layer_name, module, False)

    def __load_parameters(self, parameters, layer_name, module, is_parameter):
        for name, param in parameters:
            numpy_array = torch.clone(param).detach().numpy()
            parameter_name = layer_name + "." + name
            shared_mem_name = self.task_id + "." + layer_name + "." + name
            dst = np.ndarray(shape=self.model_structure[parameter_name]['shape'], dtype=self.model_structure[parameter_name]['dtype'],
                            buffer=self.shm_dict[shared_mem_name].buf)
            np.copyto(dst, numpy_array)
            setattr(module, name, None)

            if is_parameter:
                module.register_parameter(name, torch.nn.Parameter(torch.as_tensor(dst)))
            else:
                module.register_buffer(name, torch.as_tensor(dst))  


    def get_weights_from_shared_mem(self, end_shm_dict):
        weights_dict = OrderedDict()
        for key in self.model_structure.keys():
            numpy_array = np.ndarray(self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
                                    buffer=end_shm_dict[key].buf)
            weights_dict[key] = torch.from_numpy(numpy_array)
        return weights_dict

    def add_shm_refrence(self, end):
        temp_dict = {}
        for key in self.model_structure.keys():
            shared_mem_name = end + "." + key
            shm = SharedMemory(name=shared_mem_name)
            temp_dict[key] = shm 
        return temp_dict

    def release_share_mem(self):
        for key in self.shm_dict:
            self.shm_dict[key].close()
            self.shm_dict[key].unlink()

    def copy_weights_to_shared_memory(self, weights):
        for key in weights.keys():
            shared_mem_name = self.task_id + "." + key
            dst = np.ndarray(shape=self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
                            buffer=self.shm_dict[shared_mem_name].buf)
            src = torch.clone(weights[key]).detach().numpy()
            np.copyto(dst, src)

    def get_weights_from_shared_mem_self(self):
        weights_dict = OrderedDict()
        for key in self.model_structure.keys():
            shared_mem_name = self.task_id + "." + key
            numpy_array = np.ndarray(self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
                                    buffer=self.shm_dict[shared_mem_name].buf)
            weights_dict[key] = torch.from_numpy(numpy_array)
        return weights_dict



    