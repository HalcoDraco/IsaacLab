import numpy as np
import torch        

class BatchedPolicy:

    def __init__(self, model: torch.nn.Module, num_policies: int):
        self.model: torch.nn.Module = model
        self._model_assertions()
        self.num_policies: int = num_policies

        _named_parameters_info = [(name, param.shape, param.dtype, param.device) for name, param in self.model.named_parameters()]

        # Stores a dict with {name: shape} for each parameter in the model.
        self._list_named_params_shape: list[tuple[str, torch.Size]] = [(name, param_shape) for name, param_shape, _, _ in _named_parameters_info]

        # Store the population parameters in a named_parameters-like structure, but with an extra dimension for the population size.
        self._population_params = {name: torch.empty((num_policies, *shape), device=device, dtype=dtype) for name, shape, dtype, device in _named_parameters_info}

        # Indicates to vmap which dimension of the input has to be vectorized over.
        self._population_params_vectorize_index = {name: 0 for name, _ in self._list_named_params_shape}

        self._batched_forward = self._make_batched_forward()

    def _model_assertions(self):
        """Verify that the model has a correct structure."""
        # Check that model does not have buffers (e.g. running stats in batchnorm)
        for name, _ in self.model.named_buffers():
            raise ValueError(f"Model has buffer {name}, which is not compatible with BatchedPolicy.")
        
    def _make_batched_forward(self):

        def _policy_apply(policy_params, input):
            return torch.func.functional_call(self.model, policy_params, (input,))
        
        return torch.vmap(_policy_apply, in_dims=(self._population_params_vectorize_index, 0))
    
    def batched_forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        return self._batched_forward(self._population_params, inputs)
    
    def _unflatten_parameters(self, flat_tensors: torch.Tensor):
        """Unflatten a tensor with shape (num_policies, num_parameters)
        into self._population_params.
        """
        offset = 0
        for name, shape in self._list_named_params_shape:
            numel = torch.prod(torch.tensor(shape)).item()
            self._population_params[name] = flat_tensors[:, offset : offset + numel].reshape(self.num_policies, *shape)
            offset += numel

    @staticmethod
    def _flatten_parameters_single(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
            """Flatten a named_parameters {name: tensor} dict into a 1-D vector.
            The output shape is (num_parameters,)
            """
            return torch.cat([param.flatten() for name, param in parameters.items()])

    def _flatten_parameters_population(self) -> torch.Tensor:
        """Flatten a population of parameters into a 2-D tensor. population_params 
        is a dict {name: tensor} where each tensor has shape (num_policies, *param_shape) 
        The output shape is (num_policies, num_parameters) """
        
        return torch.vmap(self._flatten_parameters_single, in_dims=(self._population_params_vectorize_index,))(self._population_params)
    
    def _reset_model_weights(self) -> None:

        def _reset_layer_weights(layer):
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                if hasattr(layer, 'children'):
                    for child in layer.children():
                        _reset_layer_weights(child)
    
        _reset_layer_weights(self.model)

    def get_initial_random_parameters(self, n) -> torch.Tensor:
        """Initialize n random torch.nn.Module parameters
        and return them flattened as a tensor of shape (n, num_parameters).
        """
        init_params = []
        with torch.no_grad():
            for _ in range(n):
                self._reset_model_weights()
                init_params.append(self._flatten_parameters_single(dict(self.model.named_parameters()))) 

            return torch.stack(init_params, dim=0)
                    

    def set_population_parameters(self, population_params: torch.Tensor) -> None:
        """Set the parameters for the entire population.
        population_params has shape (num_policies, num_parameters)
        """
        self._unflatten_parameters(population_params)
