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

class Policy(torch.nn.Module):
    """Simple MLP policy network."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int = 1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

if __name__ == "__main__":
    torch.manual_seed(1234)

    def _count_model_params(model: torch.nn.Module) -> int:
        return sum(param.numel() for param in model.parameters())

    def _make_batched_policy(num_policies: int = 4) -> BatchedPolicy:
        model = Policy(obs_dim=3, hidden=8, action_dim=2)
        return BatchedPolicy(model=model, num_policies=num_policies)

    def test_init_and_metadata() -> None:
        bp = _make_batched_policy(num_policies=5)
        model_named_params = dict(bp.model.named_parameters())

        assert bp.num_policies == 5
        assert set(name for name, _ in bp._list_named_params_shape) == set(model_named_params.keys())
        assert set(bp._population_params.keys()) == set(model_named_params.keys())
        assert set(bp._population_params_vectorize_index.keys()) == set(model_named_params.keys())
        assert all(index == 0 for index in bp._population_params_vectorize_index.values())

        for name, model_param in model_named_params.items():
            expected_shape = (bp.num_policies, *model_param.shape)
            assert bp._population_params[name].shape == expected_shape
            assert bp._population_params[name].dtype == model_param.dtype
            assert bp._population_params[name].device == model_param.device

        assert callable(bp.batched_forward)

    def test_model_assertions_failure_on_buffers() -> None:
        class ModelWithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 2)
                self.register_buffer("buf", torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

        caught = False
        try:
            BatchedPolicy(ModelWithBuffer(), num_policies=2)
        except ValueError as exc:
            caught = "Model has buffer" in str(exc)
        assert caught

    def test_flatten_parameters_single() -> None:
        params = {
            "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "b": torch.tensor([5.0, 6.0]),
        }
        flat = BatchedPolicy._flatten_parameters_single(params)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert flat.shape == (6,)
        assert torch.equal(flat, expected)

    def test_unflatten_and_flatten_population_roundtrip() -> None:
        bp = _make_batched_policy(num_policies=3)
        num_params = _count_model_params(bp.model)
        flat = torch.randn(bp.num_policies, num_params)

        bp._unflatten_parameters(flat)

        manual_offset = 0
        for name, shape in bp._list_named_params_shape:
            numel = int(torch.prod(torch.tensor(shape)).item())
            expected_chunk = flat[:, manual_offset: manual_offset + numel].reshape(bp.num_policies, *shape)
            assert torch.equal(bp._population_params[name], expected_chunk)
            manual_offset += numel
        assert manual_offset == num_params

        reflatted = bp._flatten_parameters_population()
        assert reflatted.shape == flat.shape
        assert torch.allclose(reflatted, flat)

    def test_set_population_parameters() -> None:
        bp = _make_batched_policy(num_policies=4)
        num_params = _count_model_params(bp.model)
        new_population = torch.randn(bp.num_policies, num_params)
        bp.set_population_parameters(new_population)
        reflatted = bp._flatten_parameters_population()
        assert torch.allclose(reflatted, new_population)

    def test_make_batched_forward_matches_manual_loop() -> None:
        bp = _make_batched_policy(num_policies=4)
        num_params = _count_model_params(bp.model)
        population = torch.randn(bp.num_policies, num_params)
        bp.set_population_parameters(population)

        obs = torch.randn(bp.num_policies, 3)
        batched_out = bp.batched_forward(obs)

        manual_out = []
        for i in range(bp.num_policies):
            one_policy_params = {name: tensor[i] for name, tensor in bp._population_params.items()}
            manual_out.append(torch.func.functional_call(bp.model, one_policy_params, (obs[i],)))
        manual_out = torch.stack(manual_out, dim=0)

        assert batched_out.shape == manual_out.shape
        assert torch.allclose(batched_out, manual_out)

    def test_reset_model_weights_changes_weights() -> None:
        bp = _make_batched_policy(num_policies=2)
        with torch.no_grad():
            for param in bp.model.parameters():
                param.fill_(1.0)
            before = [param.detach().clone() for param in bp.model.parameters()]

        bp._reset_model_weights()
        after = [param.detach().clone() for param in bp.model.parameters()]

        assert any(not torch.equal(prev, cur) for prev, cur in zip(before, after, strict=True))

    def test_get_initial_random_parameters() -> None:
        bp = _make_batched_policy(num_policies=7)
        num_params = _count_model_params(bp.model)

        init = bp.get_initial_random_parameters(7)
        assert init.shape == (7, num_params)
        assert torch.isfinite(init).all()
        assert not torch.allclose(init[0], init[1])

    test_functions = [
        test_init_and_metadata,
        test_model_assertions_failure_on_buffers,
        test_flatten_parameters_single,
        test_unflatten_and_flatten_population_roundtrip,
        test_set_population_parameters,
        test_make_batched_forward_matches_manual_loop,
        test_reset_model_weights_changes_weights,
        test_get_initial_random_parameters,
    ]

    for test_fn in test_functions:
        test_fn()
        print(f"[PASS] {test_fn.__name__}")

    print(f"All tests passed ({len(test_functions)} total).")
    