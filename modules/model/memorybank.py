"""
group4 2024
"""
import torch
import torch.nn as nn
from collections import defaultdict, deque


class MemoryBank(nn.Module):
    def __init__(self, cfg):
        super(MemoryBank, self).__init__()
        # the number of representation bank queues
        self.num_attributes = cfg.DATA.NUM_ATTRIBUTES
        # input feature dim
        self.feature_dim = cfg.MODEL.EMBED_SIZE
        # the max size of each queue
        self.max_size = cfg.MODEL.MEMORYBANK.QUEUE_MAX_SIZE
        # the number of batches to update prototype bank
        self.n_batch = cfg.MODEL.MEMORYBANK.N_BATCH
        print(f"MemoryBank: {self.num_attributes} attributes, {self.max_size} samples per attribute, update every {self.n_batch} batches")
        # device
        self.device = cfg.DEVICE
        # this number is the sum of all attribute values
        self.proto_bank_size = cfg.MODEL.MEMORYBANK.PROTO_BANK_SIZE
        print(f"Prototype Bank size: {self.proto_bank_size}")
        # to judge when to update prototype bank
        self.batch_count = 0
        # momentum
        self.momentum = cfg.MODEL.MEMORYBANK.MOMENTUM
        print(f"MemoryBank momentum: {self.momentum}")
        # representation bank
        self.representation_bank = {a: deque(maxlen=self.max_size) for a in range(self.num_attributes)}
        # if you access a key that does not exist, it will be initialized with a zero tensor
        self.prototype_bank = defaultdict(lambda: defaultdict(lambda: torch.zeros(self.feature_dim, device=self.device)))

    def update_representation_bank(self, batch_features, batch_ids, batch_attributes, batch_values):
        """
        :param batch_features: [batch, embed_dim]
        :param batch_ids: [batch]
        :param batch_attributes: [batch]
        :param batch_values: [batch]
        :param momentum: f = momentum * f + (1 - momentum) * f'
        """
        for feature, sample_id, attribute, value in zip(batch_features, batch_ids, batch_attributes, batch_values):
            # Check if the sample is already in the queue
            existing_sample_index = next((index for index, (id, _, _) in enumerate(self.representation_bank[attribute]) if id == sample_id), None)
            if existing_sample_index is not None:
                # Momentum update
                # print('update, ', existing_sample_index)
                existing_feature = self.representation_bank[attribute][existing_sample_index][1]
                updated_feature = self.momentum * existing_feature + (1 - self.momentum) * feature
                self.representation_bank[attribute][existing_sample_index] = (sample_id, updated_feature, value)
            else:
                # If the queue is full, oldest item is automatically removed
                self.representation_bank[attribute].append((sample_id, feature.to(self.device), value))
         # Update Prototype Bank every n_batch
        self.batch_count += 1
        if self.batch_count % self.n_batch == 0:
            # update prototype bank
            # print('update Prototype Bank')
            self.update_prototype_bank()

    def update_prototype_bank(self):
        """
        Update prototype bank by calculating the mean of all features in the queue
        of each attribute and its value
        """
        # print(self.prototype_bank)
        for attr, queue in self.representation_bank.items():
            value_features = defaultdict(list)
            # get all same value features under one attr
            for _, feature, value in queue:
                value_features[value].append(feature)
            # calculate the mean of all features in the queue of each attribute and its value
            for value, features in value_features.items():
                self.prototype_bank[attr][value] = torch.mean(torch.stack(features), dim=0)

    def get_prototype_count(self):
        """
        Count the number of all prototypes in prototype bank
        """
        return sum(len(inner_dict) for inner_dict in self.prototype_bank.values())
    
    def get_positive_prototype_feature(self, attr, value):
        """
        Get the prototype feature of the given attribute and value
        """
        # Make sure all values -- prototypes are updated
        # If the number of prototypes is equal to the size of the prototype bank, 
        # it means that the prototype bank is full, and the prototype feature can be obtained
        if self.get_prototype_count() == self.proto_bank_size:
            return self.prototype_bank[attr][value]
        else:
            # If the prototype bank is not full, the prototype feature cannot be obtained
            return None
        
    def get_negative_prototype_feature(self, attr, value):
        """
        Negative prototype feature, which is the prototype feature of 
        all other values except the current sample value v under the same attribute a
        :param attr: attribute index
        :param value: attribute value index
        """
        if self.get_prototype_count() == self.proto_bank_size:
            # Get the prototype feature of all other values except the current sample value v under the same attribute a
            negative_prototypes = [self.prototype_bank[attr][other_value] for other_value in self.prototype_bank[attr] if other_value != value]
            # return a Tensor of shape [K_a-1, embed_dim], where K_a is the number of values under the attribute a
            return torch.stack(negative_prototypes)
        else:
            return None

    def assign_prototype_feature(self, batch_features, batch_attributes):
        """
        [Inference Stage]
        Find the most similar prototype feature under a for each sample 
        in the batch.
        :param batch_features: [batch, embed_dim]
        :param batch_attributes: [batch]
        :return: Tensor of most similar prototype features for each sample
        """
        batch_prototypes = []

        # Traverse each sample in the batch
        for feature, attribute in zip(batch_features, batch_attributes):
            if self.get_prototype_count() == self.proto_bank_size:
                # Get all prototype features (Cluster Center) under the attribute a
                prototypes = [self.prototype_bank[attribute][proto_value] for proto_value in self.prototype_bank[attribute]]
                prototypes = torch.stack(prototypes)

                # Calculate the cosine similarity between the input feature and all prototype features
                similarities = nn.functional.cosine_similarity(feature.unsqueeze(0), prototypes, dim=1)

                # Find the most similar prototype feature
                most_similar_idx = torch.argmax(similarities).item()
                # Get the most similar prototype feature
                most_similar_proto_feature = self.prototype_bank[attribute][most_similar_idx]

                batch_prototypes.append(most_similar_proto_feature)
        
        # Convert the list of tensor features to a single tensor
        batch_prototypes = torch.stack(batch_prototypes)
        return batch_prototypes

    def assign_values(self, batch_features, batch_attributes, batch_values):
        """
        [Inference Stage]
        Infer the value of each sample in the batch based on the distance 
        between the input feature and the prototype feature.
        Find the closest prototype feature and assign its value to the sample.
        :param batch_features: [batch, embed_dim]
        :param batch_attributes: [batch]
        :param batch_values: [batch]
        """
        # Record the number of samples whose values need to be updated
        updated_count = 0
        # Traverse each sample in the batch
        for i, (feature, attribute, value) in enumerate(zip(batch_features, batch_attributes, batch_values)):
            # print(feature, attribute, value)
            if self.get_prototype_count() == self.proto_bank_size:
                # print(self.prototype_bank[attribute])
                # Get all prototype features (Cluster Center) under the attribute a
                prototypes = [self.prototype_bank[attribute][proto_value] for proto_value in self.prototype_bank[attribute]]
                # print(len(prototypes))
                prototypes = torch.stack(prototypes)

                # Calculate the cosine similarity between the input feature and all prototype features
                similarities = nn.functional.cosine_similarity(feature.unsqueeze(0), prototypes, dim=1)

                # Find the most similar prototype feature
                most_similar_idx = torch.argmax(similarities).item()
                # Get the value of the most similar prototype feature
                most_similar_proto_value = list(self.prototype_bank[attribute].keys())[most_similar_idx]

                # Update v value to the value corresponding to the most similar cluster center
                if value != most_similar_proto_value:
                    # batch_values[i] = most_similar_proto_value
                    updated_count += 1
                # 直接return v
                batch_values[i] = most_similar_proto_value

        # Return the updated v value list and the number of updated samples
        return batch_values, updated_count

    def is_sample_in_queue(self, sample_id, attribute):
        """
        If the sample is in the queue, return True, otherwise return False
        :param sample_id: sample id
        :param attribute: attribute index
        """
        return any(id == sample_id for id, _, _ in self.representation_bank[attribute])

    def view_queues(self):
        """
        View the current status of the representation bank
        """
        for attr, queue in self.representation_bank.items():
            print(f"Queue for attribute {attr}:")
            for item in queue:
                print(f"  ID: {item[0]}, Value: {item[2]}")
            print("-" * 30)

    def update(self, batch_features, batch_ids, batch_attributes, batch_values):
        """
        [Training Stage] update representation bank and prototype bank, 
        user should call this function before forwarding the batch to the model
        :param batch_features: [batch, embed_dim]
        :param batch_ids: [batch]
        :param batch_attributes: [batch]
        """
        # Stop updating gradients
        with torch.no_grad():
            self.update_representation_bank(batch_features, batch_ids, batch_attributes, batch_values)

    def forward(self, batch_attributes, batch_values):
        """
        [Training Stage] get positive and negative prototype features
        :param batch_attributes: [batch]
        :param batch_values: [batch]
        :return: positive prototype features and negative prototype features
        positive prototype features: [batch, embed_dim]
        negative prototype features: a list of [K_a-1, embed_dim], len = batch
        """
        # Stop updating gradients
        with torch.no_grad():
            # positive proto batch [batch, embed_dim]
            positive_prototypes = []
            for attr, value in zip(batch_attributes, batch_values):
                positive_prototype = self.get_positive_prototype_feature(attr, value)
                # prototype feature is None when the prototype bank is not full
                # Not None when it finishes intialization
                if positive_prototype is None:
                    return None
                positive_prototypes.append(positive_prototype)
            positive_prototypes = torch.stack(positive_prototypes)
            # negative proto batch [[5 or 7 or 8 (K_a-1), embed_dim]], len = batch
            negative_prototypes = []
            for attr, value in zip(batch_attributes, batch_values):
                negative_prototype = self.get_negative_prototype_feature(attr, value)
                negative_prototypes.append(negative_prototype)

            return positive_prototypes, negative_prototypes