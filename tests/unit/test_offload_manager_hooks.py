# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OffloadModelProxy compatibility with PyTorch hooks.

This test suite validates that hooks work correctly with the OffloadModelProxy:

Critical behaviors tested:
- Forward hook registration
- Backward hook registration
- Activation capture
- Gradient manipulation
- Hook removal
- Multiple hooks
- Hooks on submodules
- Hooks through proxy vs underlying model

Hooks are essential for:
- Activation visualization
- Gradient clipping
- Feature extraction
- Custom regularization
- Debugging and monitoring
- Advanced training techniques
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from flextensor.offload_manager import OffloadConfig, OffloadManager, OffloadPhase
from flextensor.tensor_processors import create_model_with_shared_tensors


@pytest.mark.parametrize("registrations", [1, 2])
def test_ring_completion_survives_swaps_and_removal(registrations):
    original = nn.ModuleList([nn.Identity(), nn.Identity()])
    profile = create_model_with_shared_tensors(original)
    manager = OffloadManager("ring-hook-regression")
    manager._model = profile
    calls = []

    def completed(*_args):
        calls.append("complete")

    handles = [profile[-1].register_forward_hook(completed) for _ in range(registrations)]
    try:
        for root in (original, profile, original):
            manager._swap_to_new_model(root, OffloadPhase.INFERENCE)
            calls.clear()
            root[-1](torch.ones(1))
            assert len(calls) == registrations
        for remaining, handle in zip(reversed(range(registrations)), handles, strict=True):
            handle.remove()
            calls.clear()
            for root in (original, profile):
                root[-1](torch.ones(1))
            assert len(calls) == 2 * remaining
    finally:
        for handle in handles:
            handle.remove()
        manager.release()


@pytest.mark.parametrize("register_on_copy", [True, False])
def test_late_backward_hook_survives_swaps_and_removal(register_on_copy):
    original = nn.ModuleList([nn.Identity()])
    profile = create_model_with_shared_tensors(original)
    manager = OffloadManager("late-backward-hook-regression")
    manager._model = profile
    calls = []

    def double_gradient(module, grad_input, _grad_output):
        calls.append(module)
        return (grad_input[0] * 2,)

    registered = profile if register_on_copy else original
    handle = registered[0].register_full_backward_hook(double_gradient)
    try:
        for root in (original, profile, original):
            manager._swap_to_new_model(root, OffloadPhase.INFERENCE)
            calls.clear()
            x = torch.ones(1, requires_grad=True)
            root[0](x).sum().backward()
            assert calls == [root[0]]
            torch.testing.assert_close(x.grad, torch.full_like(x, 2))
        handle.remove()
        calls.clear()
        for root in (original, profile):
            x = torch.ones(1, requires_grad=True)
            root[0](x).sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
        assert calls == []
    finally:
        handle.remove()
        manager.release()


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize(
    "register,registry",
    [
        ("register_forward_pre_hook", "_forward_pre_hooks"),
        ("register_full_backward_hook", "_backward_hooks"),
        ("register_full_backward_pre_hook", "_backward_pre_hooks"),
    ],
)
def test_hook_registry_transfer_preserves_registration(shared, register, registry):
    original = nn.ModuleList([nn.Identity()])
    handle = getattr(original[0], register)(lambda *_args: None)
    target = create_model_with_shared_tensors(original) if shared else nn.ModuleList([nn.Identity()])
    manager = OffloadManager("registry-regression")
    try:
        manager._transfer_hooks(original, target)
        assert len(getattr(target[0], registry)) == 1
        handle.remove()
        assert len(getattr(original[0], registry)) == 0
        assert len(getattr(target[0], registry)) == (0 if shared else 1)
    finally:
        handle.remove()
        manager.release()


@pytest.mark.parametrize("shared", [True, False])
def test_copied_hooks_preserve_kwargs_and_always_call(shared):
    class Raising(nn.Module):
        def forward(self, *, value):
            raise ValueError(value)

    original = nn.ModuleList([Raising()])
    calls = []

    def before(_module, args, kwargs):
        calls.append(("before", args, kwargs))

    def after(_module, args, kwargs, output):
        assert output is None
        calls.append(("after", args, kwargs))

    pre = original[0].register_forward_pre_hook(before, with_kwargs=True)
    post = original[0].register_forward_hook(after, with_kwargs=True, always_call=True)
    target = create_model_with_shared_tensors(original) if shared else nn.ModuleList([Raising()])
    manager = OffloadManager("kwargs-regression")
    try:
        manager._transfer_hooks(original, target)
        with pytest.raises(ValueError, match="expected"):
            target[0](value="expected")
        assert calls == [("before", (), {"value": "expected"}), ("after", (), {"value": "expected"})]
    finally:
        pre.remove()
        post.remove()
        manager.release()


# Test models
class SimpleHookModel(nn.Module):
    """Simple model for hook testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


class MultiLayerModel(nn.Module):
    """Model with multiple named layers for hook testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        return self.layer3(x)


class TestForwardHooks:
    """Test forward hook registration and execution."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_register_forward_hook_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering a forward hook on the proxy model."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register hook
        hook_called = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_called.append(True)

        try:
            handle = proxy_model.register_forward_hook(forward_hook)
            assert handle is not None, "Should return hook handle"

            # Run forward pass
            x = torch.randn(4, 10)
            with torch.no_grad():
                _ = proxy_model(x)

            # Check hook was called
            assert len(hook_called) > 0, "Forward hook should be called"

            # Remove hook
            handle.remove()

        except Exception as e:
            pytest.fail(f"Failed to register forward hook: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_forward_hook_captures_output(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that forward hook can capture model output."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Capture output
        captured_output = []

        def capture_hook(module, input, output):  # noqa: A002
            captured_output.append(output.detach().clone())

        handle = proxy_model.register_forward_hook(capture_hook)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            actual_output = proxy_model(x)

        # Check captured output matches
        assert len(captured_output) == 1, "Should capture one output"
        assert torch.allclose(captured_output[0], actual_output), "Captured output should match actual"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_forward_hook_on_submodule(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering forward hook on a specific submodule."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register hook on specific layer
        hook_called = []

        def layer_hook(module, input, output):  # noqa: A002
            hook_called.append(output.shape)

        # Access submodule through proxy
        handle = proxy_model.layer2.register_forward_hook(layer_hook)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check hook was called with correct shape
        assert len(hook_called) == 1, "Hook should be called once"
        assert hook_called[0] == torch.Size([4, 30]), f"Expected shape (4, 30), got {hook_called[0]}"

        handle.remove()


class TestBackwardHooks:
    """Test backward hook registration and execution."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_register_backward_hook_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering a backward hook on the proxy model."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register backward hook
        hook_called = []

        def backward_hook(module, grad_input, grad_output):
            hook_called.append(True)

        try:
            handle = proxy_model.register_full_backward_hook(backward_hook)
            assert handle is not None, "Should return hook handle"

            # Run forward and backward
            x = torch.randn(4, 10, requires_grad=True)
            output = proxy_model(x)
            loss = output.sum()
            loss.backward()

            # Check hook was called
            assert len(hook_called) > 0, "Backward hook should be called"

            handle.remove()

        except Exception as e:
            pytest.fail(f"Failed to register backward hook: {e}")

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_backward_hook_gradient_clipping(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test using backward hook for gradient clipping."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register gradient clipping hook
        clipped_grads = []

        def clip_hook(module, grad_input, grad_output):
            # Capture gradient magnitudes
            for grad in grad_output:
                if grad is not None:
                    clipped_grads.append(grad.abs().max().item())

        handle = proxy_model.register_full_backward_hook(clip_hook)

        # Run forward and backward with large loss
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum() * 1000  # Large multiplier to create large gradients
        loss.backward()

        # Check gradients were captured
        assert len(clipped_grads) > 0, "Should capture gradients"

        handle.remove()


class TestActivationCapture:
    """Test activation capture use case."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_capture_activations_from_all_layers(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test capturing activations from all layers."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        # Use empty include_patterns to not wrap any modules (original behavior when module_paths=None)
        config = OffloadConfig(
            enabled=True, discovery_iters=0, profiling_iters=0, include_patterns=[], skip_discovery=False
        )

        proxy_model = om.offload(model, config=config)

        # Activation capture
        activations = {}

        def make_capture_hook(name):
            def hook(module, input, output):  # noqa: A002
                activations[name] = output.detach().clone()

            return hook

        # Register hooks on all Linear layers
        handles = []
        for name, module in proxy_model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(make_capture_hook(name))
                handles.append(handle)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check activations were captured
        assert len(activations) > 0, "Should capture activations"
        assert "layer1" in activations, "Should capture layer1 activations"
        assert "layer2" in activations, "Should capture layer2 activations"
        assert "layer3" in activations, "Should capture layer3 activations"

        # Check shapes
        assert activations["layer1"].shape == torch.Size([4, 20]), "layer1 output should be (4, 20)"
        assert activations["layer2"].shape == torch.Size([4, 30]), "layer2 output should be (4, 30)"
        assert activations["layer3"].shape == torch.Size([4, 10]), "layer3 output should be (4, 10)"

        # Cleanup
        for handle in handles:
            handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_capture_specific_layer_activations(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test capturing activations from a specific layer."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Capture only layer2
        layer2_output = []

        def capture_layer2(module, input, output):  # noqa: A002
            layer2_output.append(output.detach().clone())

        handle = proxy_model.layer2.register_forward_hook(capture_layer2)

        # Run multiple forward passes
        x1 = torch.randn(4, 10)
        x2 = torch.randn(4, 10)

        with torch.no_grad():
            _ = proxy_model(x1)
            _ = proxy_model(x2)

        # Check we captured two outputs
        assert len(layer2_output) == 2, "Should capture output from both forward passes"

        handle.remove()


class TestGradientManipulation:
    """Test gradient manipulation through hooks."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_gradient_clipping_via_hook(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test gradient clipping through backward hook."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Test gradient clipping on a specific layer
        gradient_norms = []

        def clip_gradient_hook(module, grad_input, grad_output):
            for grad in grad_output:
                if grad is not None:
                    gradient_norms.append(grad.norm().item())

        # Register on specific layer
        handle = proxy_model.fc1.register_full_backward_hook(clip_gradient_hook)

        # Run training step
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)

        output = proxy_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients were captured
        assert len(gradient_norms) > 0, "Should capture gradient norms"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_gradient_monitoring(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test monitoring gradients through hooks."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = MultiLayerModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        # Use empty include_patterns to not wrap any modules (original behavior when module_paths=None)
        config = OffloadConfig(
            enabled=True, discovery_iters=0, profiling_iters=0, include_patterns=[], skip_discovery=False
        )

        proxy_model = om.offload(model, config=config)

        # Monitor gradients for all layers
        gradient_stats = {}

        def make_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                stats = {}
                for i, grad in enumerate(grad_output):
                    if grad is not None:
                        stats[f"output_{i}"] = {
                            "mean": grad.mean().item(),
                            "std": grad.std().item(),
                            "max": grad.max().item(),
                            "min": grad.min().item(),
                        }
                gradient_stats[name] = stats

            return hook

        # Register hooks
        handles = []
        for name, module in proxy_model.named_modules():
            if isinstance(module, nn.Linear):
                handle = module.register_full_backward_hook(make_gradient_hook(name))
                handles.append(handle)

        # Training step
        x = torch.randn(4, 10, requires_grad=True)
        target = torch.randn(4, 10)

        output = proxy_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradient stats were collected
        assert len(gradient_stats) > 0, "Should collect gradient statistics"
        assert "layer1" in gradient_stats, "Should have stats for layer1"

        # Cleanup
        for handle in handles:
            handle.remove()


class TestMultipleHooks:
    """Test multiple hooks on the same module."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_multiple_forward_hooks(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test registering multiple forward hooks on the same module."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register multiple hooks
        hook1_called = []
        hook2_called = []
        hook3_called = []

        def hook1(module, input, output):  # noqa: A002
            hook1_called.append(True)

        def hook2(module, input, output):  # noqa: A002
            hook2_called.append(True)

        def hook3(module, input, output):  # noqa: A002
            hook3_called.append(True)

        handle1 = proxy_model.register_forward_hook(hook1)
        handle2 = proxy_model.register_forward_hook(hook2)
        handle3 = proxy_model.register_forward_hook(hook3)

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # All hooks should be called
        assert len(hook1_called) > 0, "Hook 1 should be called"
        assert len(hook2_called) > 0, "Hook 2 should be called"
        assert len(hook3_called) > 0, "Hook 3 should be called"

        # Cleanup
        handle1.remove()
        handle2.remove()
        handle3.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hook_execution_order(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks execute in registration order."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Track execution order
        execution_order = []

        def make_hook(hook_id):
            def hook(module, input, output):  # noqa: A002
                execution_order.append(hook_id)

            return hook

        handle1 = proxy_model.register_forward_hook(make_hook(1))
        handle2 = proxy_model.register_forward_hook(make_hook(2))
        handle3 = proxy_model.register_forward_hook(make_hook(3))

        # Run forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        # Check execution order
        assert execution_order == [1, 2, 3], f"Expected [1, 2, 3], got {execution_order}"

        # Cleanup
        handle1.remove()
        handle2.remove()
        handle3.remove()


class TestHookRemoval:
    """Test hook removal."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_remove_forward_hook(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test removing a forward hook."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register and remove hook
        hook_called = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_called.append(True)

        handle = proxy_model.register_forward_hook(forward_hook)

        # First forward pass - hook should be called
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_called) == 1, "Hook should be called once"

        # Remove hook
        handle.remove()

        # Second forward pass - hook should NOT be called
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_called) == 1, "Hook should still be called only once (not after removal)"

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_remove_one_of_multiple_hooks(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test removing one hook while others remain."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Register multiple hooks
        hook1_count = []
        hook2_count = []

        def hook1(module, input, output):  # noqa: A002
            hook1_count.append(1)

        def hook2(module, input, output):  # noqa: A002
            hook2_count.append(1)

        handle1 = proxy_model.register_forward_hook(hook1)
        handle2 = proxy_model.register_forward_hook(hook2)

        # First pass - both should be called
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook1_count) == 1, "Hook 1 should be called"
        assert len(hook2_count) == 1, "Hook 2 should be called"

        # Remove hook1
        handle1.remove()

        # Second pass - only hook2 should be called
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook1_count) == 1, "Hook 1 should not be called after removal"
        assert len(hook2_count) == 2, "Hook 2 should still be called"

        # Cleanup
        handle2.remove()


class TestHooksAcrossStateTransitions:
    """Test hook behavior across state transitions."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_persist_across_state_transitions(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks registered on model modules persist across state transitions."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Different model instances for each state (simulating model swaps)
        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()

        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=2, skip_discovery=False)

        # Offload the warmup model (initialize_warmup will return it)
        proxy_model = om.offload(warmup_model, config=config)

        # Register hook on the underlying model's fc1 layer (not the proxy)
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append(om._current_phase.name)

        # Register on the actual module, not the proxy
        handle = proxy_model.fc1.register_forward_hook(forward_hook)

        # Run through state transitions
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)  # DISCOVERY (count=1, >= 1, transition after)
            _ = proxy_model(x)  # PROFILING (count=1)
            _ = proxy_model(x)  # PROFILING (count=2, >= 2, transition after)
            _ = proxy_model(x)  # INFERENCE
            _ = proxy_model(x)  # INFERENCE

        # With automatic hook transfer, hooks should persist across transitions
        assert len(hook_calls) == 5, f"Hook should be called 5 times, got {len(hook_calls)}"
        assert hook_calls == ["DISCOVERY", "PROFILING", "PROFILING", "INFERENCE", "INFERENCE"], (
            f"Hook should be called in all states, got {hook_calls}"
        )

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_on_proxy_work_with_explicit_delegation(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks registered directly on proxy work with explicit delegation."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()

        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=2, skip_discovery=False)

        proxy_model = om.offload(warmup_model, config=config)

        # Register hook on PROXY itself (now works with explicit delegation)
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append(om._current_phase.name)

        handle = proxy_model.register_forward_hook(forward_hook)

        # Run forward passes through state transitions.  The internal state-update
        # hook (installed by offload()) is registered before user hooks, so it
        # fires first and advances the phase before the user hook reads it.  User
        # hooks therefore observe the post-transition phase.
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)  # count=1 -> transition -> PROFILING observed
            _ = proxy_model(x)  # PROFILING (count=1)
            _ = proxy_model(x)  # count=2 -> transition -> INFERENCE observed
            _ = proxy_model(x)  # INFERENCE
            _ = proxy_model(x)  # INFERENCE

        # With explicit delegation and hook transfer, hooks should work and persist
        assert len(hook_calls) == 5, f"Hook should be called 5 times, got {len(hook_calls)}"
        assert hook_calls == ["PROFILING", "PROFILING", "INFERENCE", "INFERENCE", "INFERENCE"], (
            f"Hook should observe post-transition phase, got {hook_calls}"
        )

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_all_hook_types_on_proxy(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test all hook types (forward, forward_pre, backward, backward_pre) on proxy."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Track hook calls
        hook_calls = {"forward_pre": [], "forward": [], "backward": []}

        def forward_pre_hook(module, input):  # noqa: A002
            hook_calls["forward_pre"].append("pre")
            return None

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls["forward"].append("forward")
            return None

        def backward_hook(module, grad_input, grad_output):
            hook_calls["backward"].append("backward")
            return None

        # Register all hook types on proxy
        handle_pre = proxy_model.register_forward_pre_hook(forward_pre_hook)
        handle_fwd = proxy_model.register_forward_hook(forward_hook)
        handle_bwd = proxy_model.register_full_backward_hook(backward_hook)

        # Run forward and backward pass
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum()
        loss.backward()

        # Verify all hooks were called
        assert len(hook_calls["forward_pre"]) == 1, "Forward pre-hook should be called once"
        assert len(hook_calls["forward"]) == 1, "Forward hook should be called once"
        assert len(hook_calls["backward"]) == 1, "Backward hook should be called once"

        # Clean up
        handle_pre.remove()
        handle_fwd.remove()
        handle_bwd.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_backward_pre_hook_on_proxy_if_available(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test backward pre-hook on proxy (if available in PyTorch version)."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        model = SimpleHookModel()
        mock_tensor_manager.initialize_warmup.return_value = model
        mock_tensor_manager.initialize_profile.return_value = model
        mock_tensor_manager.initialize_inference.return_value = model

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        proxy_model = om.offload(model, config=config)

        # Check if backward pre-hook is available in this PyTorch version
        if not hasattr(model, "register_full_backward_pre_hook"):
            pytest.skip("register_full_backward_pre_hook not available in this PyTorch version")

        hook_calls = []

        def backward_pre_hook(module, grad_output):
            hook_calls.append("backward_pre")
            return None

        # Register backward pre-hook on proxy
        handle = proxy_model.register_full_backward_pre_hook(backward_pre_hook)

        # Run forward and backward pass
        x = torch.randn(4, 10, requires_grad=True)
        output = proxy_model(x)
        loss = output.sum()
        loss.backward()

        # Verify hook was called
        assert len(hook_calls) == 1, "Backward pre-hook should be called once"

        handle.remove()

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_hooks_persist_across_multiple_offload_calls(self, mock_strategy_cls, mock_tensor_manager_cls):
        """Test that hooks persist when offload() is called multiple times (re-initialization)."""
        # Setup mocks
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        # Create different model instances for each offload call
        model1 = SimpleHookModel()
        model2 = SimpleHookModel()
        model2.load_state_dict(model1.state_dict())

        # First offload call uses model1
        mock_tensor_manager.initialize_warmup.return_value = model1
        mock_tensor_manager.initialize_profile.return_value = model1
        mock_tensor_manager.initialize_inference.return_value = model1

        om = OffloadManager("test")
        config = OffloadConfig(enabled=True, discovery_iters=0, profiling_iters=0, skip_discovery=False)

        # First offload call
        proxy_model = om.offload(model1, config=config)

        # Register hook on a submodule
        hook_calls = []

        def forward_hook(module, input, output):  # noqa: A002
            hook_calls.append("called")

        handle = proxy_model.fc1.register_forward_hook(forward_hook)

        # Run forward pass - hook should work
        x = torch.randn(4, 10)
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_calls) == 1, "Hook should be called once after first offload()"

        # Second offload call (re-initialization) uses model2
        mock_tensor_manager.initialize_warmup.return_value = model2

        # Call offload again - this triggers _transition_to_warmup with hook transfer
        proxy_model = om.offload(model2)

        # Run forward pass - hook should still work after re-initialization
        with torch.no_grad():
            _ = proxy_model(x)

        assert len(hook_calls) == 2, (
            f"Hook should be called twice (once before, once after re-offload), got {len(hook_calls)}"
        )

        handle.remove()


def _tagged_state_hooks(module: nn.Module) -> list:
    """Return the list of state-update hooks installed on ``module``."""
    return [h for h in module._forward_hooks.values() if getattr(h, "_ft_state_update_hook", False)]


class TestStateUpdateHookLifecycle:
    """Direct invariants of ``OffloadManager._install_state_update_hook``."""

    def test_install_is_idempotent(self):
        """Calling _install_state_update_hook twice leaves exactly one tagged hook."""
        om = OffloadManager("test_install_idempotent")
        model = SimpleHookModel()
        om._model = model

        om._install_state_update_hook()
        first_handle = om._state_hook_handle
        assert first_handle is not None
        assert len(_tagged_state_hooks(model)) == 1

        om._install_state_update_hook()
        second_handle = om._state_hook_handle

        assert second_handle is not None
        assert second_handle is not first_handle
        assert len(_tagged_state_hooks(model)) == 1

    def test_install_no_op_when_model_is_none(self):
        """If self._model is None the hook is not installed."""
        om = OffloadManager("test_install_no_model")
        om._model = None

        om._install_state_update_hook()

        assert om._state_hook_handle is None

    def test_release_clears_handle(self):
        """release() removes the tagged hook and clears self._state_hook_handle."""
        om = OffloadManager("test_release_clears")
        model = SimpleHookModel()
        om._model = model
        om._install_state_update_hook()
        assert len(_tagged_state_hooks(model)) == 1

        om.release()

        assert om._state_hook_handle is None
        assert _tagged_state_hooks(model) == []

    def test_transfer_hooks_skips_state_update_hook(self):
        """_transfer_hooks copies user hooks but never the internal state-update hook."""
        om = OffloadManager("test_transfer_skip")
        old_model = SimpleHookModel()
        new_model = SimpleHookModel()
        om._model = old_model

        # Install the state-update hook on old_model.
        om._install_state_update_hook()
        # Register a regular user hook on the same submodule we'll inspect on the new model.
        user_hook = lambda _m, _i, _o: None  # noqa: E731
        old_model.register_forward_hook(user_hook)

        om._transfer_hooks(old_model, new_model)

        new_hooks = list(new_model._forward_hooks.values())
        assert user_hook in new_hooks, "user hook must be transferred"
        assert _tagged_state_hooks(new_model) == [], "state-update hook must NOT be transferred"

    def test_transfer_hooks_preserves_with_kwargs(self):
        """Phase swaps must keep with_kwargs so hook signatures stay valid."""
        om = OffloadManager("test_transfer_kwargs")

        class KwargModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x, scale: float = 1.0):
                return self.fc(x) * scale

        old_model = KwargModel()
        new_model = KwargModel()
        seen: dict[str, object] = {}

        def pre_hook(_m, _args, kwargs):
            seen["pre"] = dict(kwargs)
            return None

        def post_hook(_m, _args, kwargs, _out):
            seen["post"] = dict(kwargs)
            return None

        old_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        old_model.register_forward_hook(post_hook, with_kwargs=True)

        om._transfer_hooks(old_model, new_model)

        # Behavioral: wrong arity if with_kwargs was dropped on re-register.
        out = new_model(torch.randn(2, 4), scale=2.0)
        assert out.shape == (2, 4)
        assert seen["pre"] == {"scale": 2.0}
        assert seen["post"] == {"scale": 2.0}

    def test_transfer_hooks_preserves_always_call(self):
        """Forward hooks registered with always_call=True must still run after transfer."""
        om = OffloadManager("test_transfer_always_call")

        class BoomModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("boom")

        old_model = BoomModel()
        new_model = BoomModel()
        called: list[str] = []

        def post_hook(_m, _args, _out):
            called.append("post")
            return None

        old_model.register_forward_hook(post_hook, always_call=True)
        om._transfer_hooks(old_model, new_model)

        post_id = next(iter(new_model._forward_hooks))
        assert new_model._forward_hooks_always_called.get(post_id) is True

        with pytest.raises(RuntimeError, match="boom"):
            new_model(torch.randn(2, 4))
        assert called == ["post"]

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_transitions_keep_handle_on_current_model(self, _strategy_cls, mock_tensor_manager_cls):
        """After each transition, the state-update hook is on the live model only."""
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()
        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test_transition_handle")
        # Explicit skip_discovery=False: this test exercises the
        # discovery -> profile -> inference state machine separately;
        # skip_discovery=True collapses discovery into profile during
        # ``offload()`` itself, which the assertions below don't cover.
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, skip_discovery=False)
        proxy = om.offload(warmup_model, config=config)

        assert om._model is warmup_model
        assert len(_tagged_state_hooks(warmup_model)) == 1
        assert om._state_hook_handle is not None

        x = torch.randn(4, 10)
        with torch.no_grad():
            proxy(x)  # discovery -> profile transition

        assert om._model is profile_model
        assert len(_tagged_state_hooks(profile_model)) == 1
        assert _tagged_state_hooks(warmup_model) == []

        with torch.no_grad():
            proxy(x)  # profile -> inference transition

        assert om._model is inference_model
        assert len(_tagged_state_hooks(inference_model)) == 1
        assert _tagged_state_hooks(profile_model) == []


class TestStateUpdateHookSafeRemoval:
    """The install/release paths must tolerate a stale handle without aborting."""

    @staticmethod
    def _break_handle(handle, exc: Exception) -> None:
        """Replace the handle's ``remove`` with one that raises ``exc``."""
        handle.remove = MagicMock(side_effect=exc)

    def test_install_warns_and_continues_when_old_handle_remove_raises(self, caplog):
        om = OffloadManager("test_install_remove_raises")
        model = SimpleHookModel()
        om._model = model
        om._install_state_update_hook()
        first_handle = om._state_hook_handle
        self._break_handle(first_handle, RuntimeError("hook dict mutated"))

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om._install_state_update_hook()

        assert om._state_hook_handle is not first_handle, (
            "new handle must be installed even when old handle removal fails"
        )
        # Old hook is leaked (we logged it instead of raising); the WARNING
        # is the user-actionable signal so a future debugger can see it.
        assert any("state-update hook handle removal failed" in r.getMessage() for r in caplog.records), (
            f"expected WARNING about handle removal; got: {[r.getMessage() for r in caplog.records]}"
        )

    def test_release_warns_and_continues_when_handle_remove_raises(self, caplog):
        om = OffloadManager("test_release_remove_raises")
        model = SimpleHookModel()
        om._model = model
        om._install_state_update_hook()
        self._break_handle(om._state_hook_handle, KeyError("already removed"))

        with caplog.at_level("WARNING", logger="flextensor.offload_manager"):
            om.release()

        assert om._state_hook_handle is None, "handle must be cleared even when remove() raises"
        assert any("state-update hook handle removal failed" in r.getMessage() for r in caplog.records), (
            f"expected WARNING about handle removal during release(); got: {[r.getMessage() for r in caplog.records]}"
        )


class TestCompileWrappedTransitionWarning:
    """Phase transitions warn when an OptimizedModule wraps the proxy."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_transition_warns_when_proxy_is_wrapped_by_torch_compile(self, _strategy_cls, mock_tensor_manager_cls):
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()
        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test_compile_wrap_warn")
        # Explicit skip_discovery=False: the test asserts the warning fires
        # on the first proxy(x) call (the discovery -> PROFILING transition);
        # skip_discovery=True does that transition inside ``offload()``,
        # before the torch.compile wrap is in place.
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, skip_discovery=False)
        proxy = om.offload(warmup_model, config=config)

        # Wrap the proxy with torch.compile (no actual call → no tracing) so
        # OptimizedModule references the proxy.
        compiled = torch.compile(proxy)
        del compiled  # only need the referrer to live; pytest holds it via warns

        x = torch.randn(4, 10)
        with torch.no_grad(), pytest.warns(RuntimeWarning, match="phase transition to PROFILING"):
            # Re-establish the OptimizedModule referrer for the duration of the call.
            wrapper = torch.compile(proxy)  # noqa: F841 — kept alive by local ref
            proxy(x)

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_transition_does_not_warn_without_compile_wrapper(self, _strategy_cls, mock_tensor_manager_cls):
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        inference_model = SimpleHookModel()
        profile_model.load_state_dict(warmup_model.state_dict())
        inference_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model
        mock_tensor_manager.initialize_inference.return_value = inference_model

        om = OffloadManager("test_compile_wrap_no_warn")
        # Explicit skip_discovery=False: see paired warn-test above.
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, skip_discovery=False)
        proxy = om.offload(warmup_model, config=config)

        x = torch.randn(4, 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                proxy(x)
                proxy(x)

        compile_warnings = [w for w in caught if "torch.compile" in str(w.message).lower()]
        assert compile_warnings == [], (
            f"no compile-wrap warning expected for un-wrapped proxy; got: {[str(w.message) for w in compile_warnings]}"
        )


class TestTransitionRollback:
    """A failed transition restores the manager to its pre-transition snapshot."""

    @patch("flextensor.tensor_manager.TensorManager")
    @patch("flextensor.strategy.KnapsackStrategy")
    def test_install_hook_failure_rolls_back_model_and_phase(self, _strategy_cls, mock_tensor_manager_cls):
        mock_tensor_manager = MagicMock()
        mock_tensor_manager.is_profiling_suspended.return_value = False
        mock_tensor_manager_cls.return_value = mock_tensor_manager
        mock_tensor_manager.trap = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        warmup_model = SimpleHookModel()
        profile_model = SimpleHookModel()
        profile_model.load_state_dict(warmup_model.state_dict())

        mock_tensor_manager.initialize_warmup.return_value = warmup_model
        mock_tensor_manager.initialize_profile.return_value = profile_model

        om = OffloadManager("test_rollback")
        # Explicit skip_discovery=False: the test asserts ``om._model is
        # warmup_model`` post-offload and rolls back on a failed
        # _transition_to_profile; skip_discovery=True collapses that
        # transition into ``offload()`` itself.
        config = OffloadConfig(enabled=True, discovery_iters=1, profiling_iters=1, skip_discovery=False)
        proxy = om.offload(warmup_model, config=config)

        assert om._model is warmup_model
        pre_phase = om._current_phase
        pre_handle = om._state_hook_handle

        with (
            patch.object(
                OffloadManager, "_install_state_update_hook", side_effect=RuntimeError("simulated install failure")
            ),
            pytest.raises(RuntimeError, match="simulated install failure"),
        ):
            om._transition_to_profile()

        assert om._model is warmup_model, "_model must roll back to the pre-transition model"
        assert om._current_phase is pre_phase, "phase must roll back"
        assert om._state_hook_handle is pre_handle, "state hook handle must roll back"
        assert om._model_proxy.__subject__ is warmup_model, "proxy subject must roll back"
        # Sanity: forward still works after rollback.
        with torch.no_grad():
            proxy(torch.randn(4, 10))


class TestStateUpdateHookPrependOrdering:
    """The internal state-update hook must be the *first* entry in
    ``_forward_hooks`` regardless of when the user registered theirs.

    This is the invariant the migration note depends on ("user-registered
    forward hooks observe the post-transition phase").  Indirect tests
    via observed phase sequences would still pass if a future refactor
    swapped to ``prepend=False`` — this pins the structural ordering.
    """

    def test_internal_hook_is_first_entry_after_user_hook_registered_first(self):
        om = OffloadManager("test_prepend_order_user_first")
        model = SimpleHookModel()

        # User registers a hook BEFORE FlexTensor takes over.
        user_hook = lambda _m, _i, _o: None  # noqa: E731
        model.register_forward_hook(user_hook)

        om._model = model
        om._install_state_update_hook()

        first_hook = next(iter(model._forward_hooks.values()))
        assert getattr(first_hook, "_ft_state_update_hook", False), (
            f"prepend=True is broken: first hook in iteration order should be the FT internal hook, got {first_hook!r}"
        )

    def test_internal_hook_is_first_entry_after_user_hook_registered_after(self):
        om = OffloadManager("test_prepend_order_user_after")
        model = SimpleHookModel()

        om._model = model
        om._install_state_update_hook()

        # User registers a hook AFTER FlexTensor — must still observe post-transition phase.
        user_hook = lambda _m, _i, _o: None  # noqa: E731
        model.register_forward_hook(user_hook)

        first_hook = next(iter(model._forward_hooks.values()))
        assert getattr(first_hook, "_ft_state_update_hook", False), (
            f"prepend=True is broken: first hook should remain FT internal even after user hook is added, "
            f"got {first_hook!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
