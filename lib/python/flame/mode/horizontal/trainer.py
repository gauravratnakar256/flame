# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""horizontal FL trainer."""

import logging
import time
import torch
import numpy as np
from collections import OrderedDict
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import shared_memory

from ...channel_manager import ChannelManager
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ...common.util import (MLFramework, get_ml_framework_in_use,
                            mlflow_runname, valid_frameworks)
from ...registries import registry_provider
from ..composer import Composer
from ..message import MessageType
from ..role import Role
from ..tasklet import Loop, Tasklet
from multiprocessing import resource_tracker



logger = logging.getLogger(__name__)

TAG_FETCH = 'fetch'
TAG_UPLOAD = 'upload'

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


class Trainer(Role, metaclass=ABCMeta):
    """Trainer implements an ML training role."""

    @abstract_attribute
    def config(self):
        """Abstract attribute for config object."""

    @abstract_attribute
    def dataset_size(self):
        """Abstract attribute for size of dataset used to train."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.registry_client = registry_provider.get(self.config.registry.sort)
        # initialize registry client
        self.registry_client(self.config.registry.uri, self.config.job.job_id)

        self.registry_client.setup_run(mlflow_runname(self.config))
        self.metrics = dict()

        self._round = 1
        self._work_done = False

        self.shm_dict_list = {}

        self.shm_dict = {}
        self.model_structure = OrderedDict()
        self.task_id = self.config.task_id

        self.framework = get_ml_framework_in_use()
        if self.framework == MLFramework.UNKNOWN:
            raise NotImplementedError(
                "supported ml framework not found; "
                f"supported frameworks are: {valid_frameworks}")

    def create_structure(self, parameters, layer_name):
        for name, param in parameters:
             numpy_array = torch.clone(param).detach().numpy()
             numpy_array_datatype = numpy_array.dtype
             mem_size = int(numpy_array.nbytes)
             parameter_name =  layer_name + "." + name
             shared_mem_name = self.task_id + "." + layer_name + "." + name
             remove_shm_from_resource_tracker()
             shm = shared_memory.SharedMemory(name=shared_mem_name, create=True, size=mem_size)
             self.shm_dict[shared_mem_name] = shm
             self.model_structure[parameter_name] = {'memsize': mem_size, 'dtype': numpy_array_datatype,'shape': numpy_array.shape}

    def create_model_structure(self):
        for layer_name, module in self.model.named_modules():
            self.create_structure(module.named_parameters(recurse=False), layer_name)
            self.create_structure(module.named_buffers(recurse=False), layer_name)

    def load_parameters_to_shared_memory(self):
        for layer_name, module in self.model.named_modules():
            self.load_parameters(module.named_parameters(recurse=False), layer_name)
            self.load_parameters(module.named_buffers(recurse=False), layer_name)

    def load_parameters(self, parameters, layer_name):
        for name, param in parameters:
            numpy_array = torch.clone(param).detach().numpy()
            parameter_name = layer_name + "." + name
            shared_mem_name = self.task_id + "." + layer_name + "." + name
            dst = np.ndarray(shape=self.model_structure[parameter_name]['shape'], dtype=self.model_structure[parameter_name]['dtype'],
                            buffer=self.shm_dict[shared_mem_name].buf)
            np.copyto(dst, numpy_array)


    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)

    # def get_weights_from_shared_memory(self, temp_dict):
    #     model_dict = {}
    #     for key in temp_dict:
    #         parameter_name = "trainer" + "." + key
    #         numpy_array = np.ndarray(self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
    #                                 buffer=self.temp_Dict[key].buf)
    #         self.model_dict[key] = torch.from_numpy(numpy_array)

    def add_shm_refrence(self, end):
        temp_dict = {}
        for key in self.model_structure.keys():
            shared_mem_name = end + "." + key
            shm = SharedMemory(name=shared_mem_name)
            temp_dict[key] = shm 
        return temp_dict

    def get_weights_from_shared_mem(self, end):
        weights_dict = OrderedDict()
        for key in self.model_structure.keys():
            numpy_array = np.ndarray(self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
                                    buffer=self.shm_dict_list[end][key].buf)
            weights_dict[key] = torch.from_numpy(numpy_array)
        return weights_dict


    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return
        
        # this call waits for at least one peer joins this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()
        msg = channel.recv(end)

        if not end in self.shm_dict_list:
            temp_dict = self.add_shm_refrence(end)
            self.shm_dict_list[end] = temp_dict

        weights = self.get_weights_from_shared_mem(end)
        
        # logger.info("The end id of aggregator is " + end)

        if MessageType.WEIGHTS in msg:
            #self.weights = msg[MessageType.WEIGHTS]
            self.weights = weights
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

        logger.info(self._round)

        logger.debug(f"work_done: {self._work_done}, round: {self._round}")

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)

    def _send_weights(self, tag: str) -> None:
        logger.debug("calling _send_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_send_weights] channel not found with {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()

        self._update_weights()
        channel.send(
            end, {
                MessageType.WEIGHTS: self.weights,
                MessageType.DATASET_SIZE: self.dataset_size
            })
        logger.debug("sending weights done")

    def save_metrics(self):
        """Save metrics in a model registry."""
        logger.debug(f"saving metrics: {self.metrics}")
        if self.metrics:
            self.registry_client.save_metrics(self._round - 1, self.metrics)
            logger.debug("saving metrics done")

    def update_metrics(self, metrics: dict[str, float]):
        """Update metrics."""
        self.metrics = self.metrics | metrics

    def _update_model(self):
        if self.framework == MLFramework.PYTORCH:
            self.model.load_state_dict(self.weights)
        elif self.framework == MLFramework.TENSORFLOW:
            self.model.set_weights(self.weights)

    def _update_weights(self):
        if self.framework == MLFramework.PYTORCH:
            self.weights = self.model.state_dict()
        elif self.framework == MLFramework.TENSORFLOW:
            self.weights = self.model.get_weights()

    def release_share_mem(self):
        del self.shm_dict_list
        for key in self.shm_dict:
            self.shm_dict[key].close()
            self.shm_dict[key].unlink()

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_load_data = Tasklet(self.load_data)

            task_init = Tasklet(self.initialize)

            task_create_model_structure = Tasklet(self.create_model_structure)

            task_get = Tasklet(self.get, TAG_FETCH)

            task_train = Tasklet(self.train)

            task_eval = Tasklet(self.evaluate)

            task_put = Tasklet(self.put, TAG_UPLOAD)

            task_save_metrics = Tasklet(self.save_metrics)

            task_release_share_mem = Tasklet(self.release_share_mem)

            # create a loop object with loop exit condition function
            loop = Loop(loop_check_fn=lambda: self._work_done)
            task_internal_init >> task_load_data >> task_init >> task_create_model_structure >> loop(
                task_get >> task_train >> task_eval >> task_put >>
                task_save_metrics ) >> task_release_share_mem

            logger.info("Done with training")

    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the trainer role."""
        return [TAG_FETCH, TAG_UPLOAD]
