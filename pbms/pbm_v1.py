
from typing import List
import random

import torch

from .pbm import PBMBase, kernel_index, new_kernel


class PBM_v1(PBMBase):


    def _l2_distance(self, tensor1, tensor2):
        return torch.norm(tensor1 - tensor2, p=2)


    def _calculate_entropy(self, tensor):
        tensor_flat = tensor.view(-1)
        
        tensor_flat = (tensor_flat - tensor_flat.min()) / (tensor_flat.max() - tensor_flat.min() + 1e-12)
        
        tensor_prob = tensor_flat / torch.sum(tensor_flat)
        
        entropy = -torch.sum(tensor_prob * torch.log2(tensor_prob + 1e-12)) 
        
        return entropy.item()

    def _geometric_median_distances(self,tensors):
        num_tensors = len(tensors)
        distances = torch.zeros(num_tensors, num_tensors)

        for i in range(num_tensors):
            for j in range(num_tensors):
                if i != j:
                    distances[i, j] = self._l2_distance(tensors[i], tensors[j])

        return distances

    

    def _k_medoids(self, distances, n_groups, max_iter=30):
        n_kernels = distances.shape[0]
        k = n_groups

        medoids = random.sample(range(0, n_kernels), k)
        clusters = None

        for _ in range(max_iter):
            clusters = [[] for _ in range(k)]
            for kernel_id in range(n_kernels):
                closest_medoid = min(medoids, key=lambda m_ker_id: distances[kernel_id][m_ker_id])
                clusters[medoids.index(closest_medoid)].append(kernel_id)

            clusters = [cluster for cluster in clusters if cluster]
            

            if not clusters:
                break

                
            new_medoids = []
            for cluster in clusters:

                cluster_distances = [sum(distances[i][j] for j in cluster) for i in cluster]

                min_distance_index = cluster_distances.index(min(cluster_distances))
                new_medoids.append(cluster[min_distance_index])

            if set(medoids) == set(new_medoids):
                break
            medoids = new_medoids
        return clusters




    
    def _calculate_kernel_rating(self, kernel: torch.Tensor)->float:
        # input is 3d tensor
        return self._calculate_entropy(kernel)



    def _group_kernels_together(self, layer_weights_tensor: torch.Tensor, indexes_of_depleated_kernels:List[kernel_index], number_of_groups:int)->List[List[kernel_index]]:
        distances = self._geometric_median_distances(layer_weights_tensor[indexes_of_depleated_kernels])
        return self._k_medoids(distances,number_of_groups)



    def _merge_tensor(self, tensor_to_be_merged:torch.Tensor)->new_kernel:
        # input is 4d and out is 3d
        return torch.mean(tensor_to_be_merged , dim=0)
        
