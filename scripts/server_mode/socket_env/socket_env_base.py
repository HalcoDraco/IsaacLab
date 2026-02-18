import socket
import torch

class SocketEnv:

    DEFAULT_SOCKET_PATH = "/tmp/sockets/isaac_communication.sock"

    STEP = b"\x01"
    RESET = b"\x02"
    CLOSE = b"\x03"
    MAKE = b"\x04"
    STOP = b"\x05"

    def __init__(self, socket_path: str | None, device: str = "cuda", disable_fabric: bool = False):
        if socket_path is None:
            socket_path = self.DEFAULT_SOCKET_PATH
        self._socket_path = socket_path
        self.device = device
        self.disable_fabric = disable_fabric

        self.obs_buffer: torch.Tensor | None = None
        self.rewards_buffer: torch.Tensor | None = None
        self.terminated_buffer: torch.Tensor | None = None
        self.truncated_buffer: torch.Tensor | None = None
        self.action_buffer: torch.Tensor | None = None

    @staticmethod
    def _get_tensor_metadata(tensor: torch.Tensor) -> dict:
        """Extract metadata from a CUDA tensor for IPC sharing."""
        if not tensor.is_cuda:
            raise ValueError("Only CUDA tensors can be shared via IPC.")
        
        ipc_keys = [
            "storage_device", 
            "storage_handle", 
            "storage_size_bytes",
            "storage_offset_bytes", 
            "ref_counter_handle", 
            "ref_counter_offset",
            "event_handle", 
            "event_sync_required",
        ]
        meta = {
            "tensor_cls": type(tensor),
            "tensor_size": tensor.shape,
            "tensor_stride": tensor.stride(),
            "tensor_offset": tensor.storage_offset(),
            "storage_cls": torch.UntypedStorage,
            "dtype": tensor.dtype,
            "requires_grad": tensor.requires_grad,
            **dict(zip(ipc_keys, tensor.untyped_storage()._share_cuda_())),
        }
        return meta

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
        """Receive exactly ``num_bytes`` bytes from a socket."""
        chunks = bytearray()
        while len(chunks) < num_bytes:
            chunk = sock.recv(num_bytes - len(chunks))
            if not chunk:
                raise RuntimeError("Socket connection closed while receiving data.")
            chunks.extend(chunk)
        return bytes(chunks)