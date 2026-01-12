"""Type stub overrides for moderngl to fix .value attribute typing."""

from __future__ import annotations

from typing import Any, Optional, Union, Tuple

class Buffer:
    """ModernGL buffer object."""

    size: int
    def write(self, data: Any, offset: int = 0) -> None: ...
    def read(self, size: int = -1, offset: int = 0) -> bytes: ...
    def release(self) -> None: ...
    def bind_to_storage_buffer(self, binding: int) -> None: ...

class ComputeShader:
    """ModernGL compute shader program."""

    def __getitem__(self, key: str) -> Uniform: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def run(self, group_x: int = 1, group_y: int = 1, group_z: int = 1) -> None: ...
    def release(self) -> None: ...

class Context:
    """ModernGL rendering context."""

    info: dict[str, str]
    def buffer(
        self, data: Any = None, reserve: int = 0, dynamic: bool = False
    ) -> Buffer: ...
    def texture(
        self,
        size: tuple[int, int],
        components: int = 4,
        data: Optional[Any] = None,
        dtype: str = "f4",
        alignment: int = 1,
        samples: int = 0,
    ) -> Texture: ...
    def compute_shader(self, source: str) -> ComputeShader: ...
    def release(self) -> None: ...
    def clear(
        self,
        red: float = 0.0,
        green: float = 0.0,
        blue: float = 0.0,
        alpha: float = 0.0,
        depth: float = 1.0,
        viewport: Optional[Tuple[int, int, int, int]] = None,
    ) -> None: ...
    def memory_barrier(self, barriers: int) -> None: ...

class Texture:
    """ModernGL texture object."""

    repeat_x: bool
    repeat_y: bool
    def bind_to_image(
        self, unit: int, read: bool = True, write: bool = False, level: int = 0
    ) -> None: ...
    def use(self, location: int = 0) -> None: ...
    def read_into(self, buffer: Buffer, level: int = 0, alignment: int = 1) -> None: ...
    def release(self) -> None: ...

class Uniform:
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, value: Union[int, float, bool, tuple, list]) -> None: ...

class UniformBlock:
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, value: Union[int, float, bool, tuple, list]) -> None: ...

class StorageBlock:
    @property
    def value(self) -> Any: ...
    @value.setter
    def value(self, value: Union[int, float, bool, tuple, list]) -> None: ...

def create_standalone_context(
    require: Optional[int] = None, **kwargs: Any
) -> Context: ...
def create_context(require: Optional[int] = None, **kwargs: Any) -> Context: ...

# OpenGL memory barrier bits
SHADER_IMAGE_ACCESS_BARRIER_BIT: int
TEXTURE_FETCH_BARRIER_BIT: int
