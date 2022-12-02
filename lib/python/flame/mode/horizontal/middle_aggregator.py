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
"""honrizontal FL middle level aggregator."""

import logging
import time
import torch
import numpy as np
from collections import OrderedDict

from diskcache import Cache

from ...channel_manager import ChannelManager
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from ...common.custom_abcmeta import ABCMeta, abstract_attribute
from ...common.util import (MLFramework, get_ml_framework_in_use, valid_frameworks)
from ...optimizer.train_result import TrainResult
from ...optimizers import optimizer_provider
from ...plugin import PluginManager
from ..composer import Composer
from ..message import MessageType
from ..role import Role
from ..tasklet import Loop, Tasklet
from multiprocessing import resource_tracker

logger = logging.getLogger(__name__)

TAG_DISTRIBUTE = 'distribute'
TAG_AGGREGATE = 'aggregate'
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


class MiddleAggregator(Role, metaclass=ABCMeta):
    """Middle level aggregator.

    It acts as a proxy between top level aggregator and trainer.
    """

    @abstract_attribute
    def config(self):
        """Abstract attribute for config object."""

    def internal_init(self) -> None:
        """Initialize internal state for role."""
        # global variable for plugin manager
        self.plugin_manager = PluginManager()

        self.cm = ChannelManager()
        self.cm(self.config)
        self.cm.join_all()

        self.shm_dict = {}
        self.model_structure = OrderedDict()
        self.task_id = self.config.task_id
        self.shm_dict_list = {}

        self.optimizer = optimizer_provider.get(self.config.optimizer.sort,
                                                **self.config.optimizer.kwargs)

        self._round = 1
        self._work_done = False

        self.cache = Cache()
        self.dataset_size = 0

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

    def get_weights_from_shared_mem(self, end):
        weights_dict = OrderedDict()
        for key in self.model_structure.keys():
            numpy_array = np.ndarray(self.model_structure[key]['shape'], dtype=self.model_structure[key]['dtype'],
                                    buffer=self.shm_dict_list[end][key].buf)
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

    def get(self, tag: str) -> None:
        """Get data from remote role(s)."""
        if tag == TAG_FETCH:
            self._fetch_weights(tag)
        if tag == TAG_AGGREGATE:
            self._aggregate_weights(tag)

    def put(self, tag: str) -> None:
        """Set data to remote role(s)."""
        if tag == TAG_UPLOAD:
            self._send_weights(tag)
        if tag == TAG_DISTRIBUTE:
            self.dist_tag = tag
            self._distribute_weights(tag)

    def _fetch_weights(self, tag: str) -> None:
        logger.debug("calling _fetch_weights")
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"[_fetch_weights] channel not found with tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        # one aggregator is sufficient
        end = channel.one_end()
        msg = channel.recv(end)

        if not end in self.shm_dict_list:
            temp_dict = self.add_shm_refrence(end)
            self.shm_dict_list[end] = temp_dict

        if MessageType.WEIGHTS in msg:
            self.weights = self.get_weights_from_shared_mem(end)
            self._update_model()

        if MessageType.EOT in msg:
            self._work_done = msg[MessageType.EOT]

        if MessageType.ROUND in msg:
            self._round = msg[MessageType.ROUND]

    def _distribute_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            logger.debug(f"channel not found for tag {tag}")
            return

        # this call waits for at least one peer to join this channel
        channel.await_join()

        self._update_weights()

        #self.load_parameters_to_shared_memory()

        for end in channel.ends():
            logger.debug(f"sending weights to {end}")
            channel.send(end, {
                MessageType.WEIGHTS: self.weights,
                MessageType.ROUND: self._round
            })

    def _aggregate_weights(self, tag: str) -> None:
        channel = self.cm.get_by_tag(tag)
        if not channel:
            return

        total = 0
        # receive local model parameters from trainers
        for end, msg in channel.recv_fifo(channel.ends()):
            if not msg:
                logger.debug(f"No data from {end}; skipping it")
                continue

            if MessageType.WEIGHTS in msg:
                weights = msg[MessageType.WEIGHTS]

            if MessageType.DATASET_SIZE in msg:
                count = msg[MessageType.DATASET_SIZE]
                total += count

            logger.debug(f"{end}'s parameters trained with {count} samples")

            tres = TrainResult(weights, count)
            # save training result from trainer in a disk cache
            self.cache[end] = tres

        # optimizer conducts optimization (in this case, aggregation)
        global_weights = self.optimizer.do(self.cache, total)
        if global_weights is None:
            logger.debug("failed model aggregation")
            time.sleep(1)
            return

        # set global weights
        self.weights = global_weights
        self.dataset_size = total

        self._update_model()


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

        self.load_parameters_to_shared_memory()

        self._update_weights()

        channel.send(
            end, {
                MessageType.WEIGHTS: self.weights,
                MessageType.DATASET_SIZE: self.dataset_size
            })
        logger.debug("sending weights done")

    def update_round(self):
        """Update the round counter."""
        logger.debug(f"Update current round: {self._round}")

        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        # set necessary properties to help channel decide how to select ends
        channel.set_property("round", self._round)

    def inform_end_of_training(self) -> None:
        """Inform all the trainers that the training is finished."""
        channel = self.cm.get_by_tag(self.dist_tag)
        if not channel:
            logger.debug(f"channel not found for tag {self.dist_tag}")
            return

        channel.broadcast({MessageType.EOT: self._work_done})

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

    def compose(self) -> None:
        """Compose role with tasklets."""
        with Composer() as composer:
            self.composer = composer

            task_internal_init = Tasklet(self.internal_init)

            task_init = Tasklet(self.initialize)

            task_load_data = Tasklet(self.load_data)

            task_create_model_structure = Tasklet(self.create_model_structure)

            task_put_dist = Tasklet(self.put, TAG_DISTRIBUTE)

            task_put_upload = Tasklet(self.put, TAG_UPLOAD)

            task_get_aggr = Tasklet(self.get, TAG_AGGREGATE)

            task_get_fetch = Tasklet(self.get, TAG_FETCH)

            task_eval = Tasklet(self.evaluate)

            task_update_round = Tasklet(self.update_round)

            task_end_of_training = Tasklet(self.inform_end_of_training)

            task_release_share_mem = Tasklet(self.release_share_mem)

        # create a loop object with loop exit condition function
        loop = Loop(loop_check_fn=lambda: self._work_done)
        task_internal_init >> task_load_data >> task_init >> task_create_model_structure >>loop(
            task_get_fetch >> task_put_dist >> task_get_aggr >> task_put_upload
            >> task_eval >> task_update_round) >> task_end_of_training >> task_release_share_mem


    def run(self) -> None:
        """Run role."""
        self.composer.run()

    @classmethod
    def get_func_tags(cls) -> list[str]:
        """Return a list of function tags defined in the middle level aggregator role."""
        return [TAG_DISTRIBUTE, TAG_AGGREGATE, TAG_FETCH, TAG_UPLOAD]
