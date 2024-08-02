
from abc import ABC, abstractmethod
from typing import List

import torch

kernel_index = int
new_kernel = torch.Tensor



class PBMBase(ABC):



    @abstractmethod
    def _calculate_kernel_rating(self, kernel: torch.Tensor)->float:
        pass


    @abstractmethod
    def _group_kernels_together(self, layer_weights_tensor: torch.Tensor, indexes_of_depleated_kernels:List[kernel_index], number_of_groups:int)->List[List[kernel_index]]:
        pass


    @abstractmethod
    def _merge_tensor(self, tensor_to_be_merged:torch.Tensor)->new_kernel:
        pass
        







    def _select_depleated_kernels(self, layer_weights_tensor: torch.Tensor, number_of_kernels_to_be_selected)->List[kernel_index]:
        number_of_kernels = layer_weights_tensor.shape[0]
        ratings = []
        for kernel_id in range(number_of_kernels):
            rating = self._calculate_kernel_rating(layer_weights_tensor[kernel_id])
            ratings += [(rating, kernel_id)]

        ratings.sort()

        selected_ids = [kernel_desc_tuple[1] for kernel_desc_tuple in ratings[:number_of_kernels_to_be_selected]]
        
        return selected_ids




    def _zero_kernel(self, current_layer:torch.nn.modules.conv.Conv2d, next_layer:torch.nn.modules.conv.Conv2d, kernel_id, batch_norm_layer = None):
        current_layer.weight.data[kernel_id].zero_()
        next_layer.weight.data[:, kernel_id, :, :].zero_()
        with torch.no_grad():
            current_layer.bias.data[kernel_id].zero_()

            if batch_norm_layer is not None:
                if batch_norm_layer.running_mean is not None:
                    batch_norm_layer.running_mean[kernel_id] = 0
                if batch_norm_layer.running_var is not None:
                    batch_norm_layer.running_var[kernel_id] = 0
                if batch_norm_layer.weight is not None:
                    batch_norm_layer.weight[kernel_id] = 0
                if batch_norm_layer.bias is not None:
                    batch_norm_layer.bias[kernel_id] = 0




    def _merge_grouped_kernels(self, current_layer:torch.nn.modules.conv.Conv2d, next_layer:torch.nn.modules.conv.Conv2d, groups_of_kernels:List[List[kernel_index]], batch_norm_layer=None)->None:
        
        for kernel_group_ids in groups_of_kernels:
            group_of_tensors = current_layer.weight.data[kernel_group_ids] # 4D tensor [kernel, input_chanel, size, size]
            new_kernel = self._merge_tensor(group_of_tensors)
            id_of_the_new_kernel = kernel_group_ids[0]
            for kernel_id in kernel_group_ids[1:]:
                self._zero_kernel(current_layer,next_layer, kernel_id, batch_norm_layer)

            # now put `new_kernel` into the  kernel with id `id_of_the_new_kernel`
            current_layer.weight.data[id_of_the_new_kernel] = new_kernel

        



    def run_pruning_by_merging(self,current_layer:torch.nn.modules.conv.Conv2d, next_layer:torch.nn.modules.conv.Conv2d, fraction_of_kernels_to_prune:float, reduction_factor:float, batch_norm_layer=None):
        
        print(f"[PBM] Starting")

        number_of_kernels = current_layer.weight.data.shape[0]
        number_of_kernels_to_be_selected = round(number_of_kernels*fraction_of_kernels_to_prune)
        number_of_groups = number_of_kernels_to_be_selected//reduction_factor
        number_of_groups = max(number_of_groups, 1)

        print(f"[PBM] Number of kernels in the layer: {number_of_kernels}")
        print(f"[PBM] Procentage of kernels to be selected: {fraction_of_kernels_to_prune*100}%")
        print(f"[PBM] Number of kernels considered: {number_of_kernels_to_be_selected}")
        print(f"[PBM] Compresion factor: {reduction_factor}")
        print(f"[PBM] Maximal umber of groups: {number_of_groups}")

        # Selecting depleated kernels
        ids_of_depleated_kernels = self._select_depleated_kernels(current_layer.weight.data,number_of_kernels_to_be_selected)
        print(f"[PBM] Kernels sellected: {str(ids_of_depleated_kernels)}")
        # List of lists containing Ids of kernels in a single group
        groups_of_kernels: List[List[int]] = self._group_kernels_together(current_layer.weight.data, ids_of_depleated_kernels, number_of_groups)
        print(f"[PBM] Group assignemnt {str(groups_of_kernels)}")

        # Merging selected kernels
        self._merge_grouped_kernels(current_layer,next_layer,groups_of_kernels, batch_norm_layer)

        

        
        
        
        print(f"[PBM] End")