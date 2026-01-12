"""
UV-texture-based two-pass GPU renderer using OpenGL compute shaders.

Pass A: Build 2D "ink SDF" texture in UV space
Pass B: Ray march torus surface and sample ink texture

This approach scales O(pixels) instead of O(pixels × primitives), enabling
rendering of thousands of nodes/edges efficiently.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import moderngl

class UVGPURenderer:
    """
    Two-pass OpenGL compute shader renderer:
    1. Build ink SDF texture in UV space
    2. Ray march torus + sample texture for coloring
    
    Scales to thousands of nodes/edges without performance degradation.
    """
    
    def __init__(self, width: int, height: int, uv_texture_size: int = 2048):
        """
        Initialize OpenGL context and compile shaders.
        
        Args:
            width: Output image width in pixels
            height: Output image height in pixels
            uv_texture_size: Resolution of UV ink SDF texture (e.g., 2048x2048)
        """
        self.width = width
        self.height = height
        self.uv_texture_size = uv_texture_size
        
        # Create OpenGL context (standalone for headless rendering)
        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenGL context: {e}")
        
        # Get GPU info
        self.gpu_info = self._get_gpu_info()
        
        # Load and compile shaders
        self.ink_sdf_program = self._compile_shader('ink_sdf.comp')
        self.raymarch_program = self._compile_shader('raymarch_uv.comp')
        
        # Create UV ink SDF texture (R32F format for signed distance)
        self.ink_sdf_texture = self.ctx.texture(
            (uv_texture_size, uv_texture_size),
            components=1,
            dtype='f4'
        )
        
        # Create output image texture (RGBA32F)
        self.output_texture = self.ctx.texture(
            (width, height),
            components=4,
            dtype='f4'
        )
        
        # Buffers (will be populated later)
        self.node_uv_buffer = None
        self.node_radii_buffer = None
        self.edge_uv_buffer = None
        self.edge_radii_buffer = None
        self.output_buffer = None
        
        # Current scene parameters
        self.num_nodes = 0
        self.num_edges = 0
        
    def _get_gpu_info(self) -> dict:
        """Get GPU information from OpenGL context."""
        return {
            'vendor': self.ctx.info.get('GL_VENDOR', 'Unknown'),
            'renderer': self.ctx.info.get('GL_RENDERER', 'Unknown'),
            'version': self.ctx.info.get('GL_VERSION', 'Unknown'),
            'glsl_version': self.ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown'),
        }
    
    def _compile_shader(self, shader_name: str) -> moderngl.ComputeShader:
        """Load and compile a compute shader."""
        shader_path = Path(__file__).parent.parent / 'shaders' / shader_name
        
        try:
            with open(shader_path, 'r') as f:
                shader_source = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Compute shader not found at {shader_path}")
        
        try:
            program = self.ctx.compute_shader(shader_source)
        except Exception as e:
            raise RuntimeError(f"Failed to compile {shader_name}: {e}")
        
        return program
    
    def upload_scene(
        self,
        node_uv: np.ndarray,          # [N, 2] - (u, v) coordinates in [-PI, PI]
        node_radii_uv: np.ndarray,    # [N] - Dot radii in UV space
        edge_uv: np.ndarray,          # [E, 4] - (u0, v0, u1, v1) for each edge
        edge_radii_uv: np.ndarray,    # [E] - Line radii in UV space
        torus_R: float,               # Major radius
        torus_r: float,               # Minor radius
        smooth_k: float,              # Smoothing factor
        ink_threshold: float,         # SDF threshold for ink visibility
        ink_smoothness: float,        # Smoothness of ink edge
        hit_eps: float,
        t_max: float,
        max_steps: int,
        camera_pos: np.ndarray,
        camera_right: np.ndarray,
        camera_up: np.ndarray,
        camera_forward: np.ndarray,
        light_dir: np.ndarray,
        ambient: float,
        diffuse_strength: float,
        specular_strength: float,
        shininess: float,
        ink_color: np.ndarray,        # [4] - RGBA
        torus_color: np.ndarray,      # [4] - RGBA
        background_color: np.ndarray, # [4] - RGBA
    ):
        """
        Upload scene data to GPU and set uniforms.
        
        This prepares both Pass A (ink SDF) and Pass B (ray march + sample).
        """
        self.num_nodes = len(node_uv)
        self.num_edges = len(edge_uv)
        
        # Ensure data is float32
        node_uv = np.ascontiguousarray(node_uv, dtype=np.float32)
        node_radii_uv = np.ascontiguousarray(node_radii_uv, dtype=np.float32)
        edge_uv = np.ascontiguousarray(edge_uv, dtype=np.float32)
        edge_radii_uv = np.ascontiguousarray(edge_radii_uv, dtype=np.float32)
        
        # Create or update buffers for Pass A (ink SDF generation)
        if (self.node_uv_buffer is None or 
            self.node_uv_buffer.size != node_uv.nbytes):
            if self.node_uv_buffer is not None:
                self.node_uv_buffer.release()
            self.node_uv_buffer = self.ctx.buffer(node_uv.tobytes())
        else:
            self.node_uv_buffer.write(node_uv.tobytes())
        
        if (self.node_radii_buffer is None or 
            self.node_radii_buffer.size != node_radii_uv.nbytes):
            if self.node_radii_buffer is not None:
                self.node_radii_buffer.release()
            self.node_radii_buffer = self.ctx.buffer(node_radii_uv.tobytes())
        else:
            self.node_radii_buffer.write(node_radii_uv.tobytes())
        
        if (self.edge_uv_buffer is None or 
            self.edge_uv_buffer.size != edge_uv.nbytes):
            if self.edge_uv_buffer is not None:
                self.edge_uv_buffer.release()
            self.edge_uv_buffer = self.ctx.buffer(edge_uv.tobytes())
        else:
            self.edge_uv_buffer.write(edge_uv.tobytes())
        
        if (self.edge_radii_buffer is None or 
            self.edge_radii_buffer.size != edge_radii_uv.nbytes):
            if self.edge_radii_buffer is not None:
                self.edge_radii_buffer.release()
            self.edge_radii_buffer = self.ctx.buffer(edge_radii_uv.tobytes())
        else:
            self.edge_radii_buffer.write(edge_radii_uv.tobytes())
        
        # Create output buffer if needed
        if self.output_buffer is None:
            output_size = self.width * self.height * 4 * 4  # RGBA float32
            self.output_buffer = self.ctx.buffer(reserve=output_size)
        
        # Set uniforms for Pass A (ink SDF)
        self.ink_sdf_program['numNodes'] = self.num_nodes
        self.ink_sdf_program['numEdges'] = self.num_edges
        self.ink_sdf_program['torusR'] = torus_R
        self.ink_sdf_program['torusr'] = torus_r
        self.ink_sdf_program['smoothK'] = smooth_k
        self.ink_sdf_program['texWidth'] = self.uv_texture_size
        self.ink_sdf_program['texHeight'] = self.uv_texture_size
        
        # Set uniforms for Pass B (ray march + sample)
        self.raymarch_program['screenWidth'] = self.width
        self.raymarch_program['screenHeight'] = self.height
        self.raymarch_program['torusR'] = torus_R
        self.raymarch_program['torusr'] = torus_r
        self.raymarch_program['inkThreshold'] = ink_threshold
        self.raymarch_program['inkSmoothness'] = ink_smoothness
        self.raymarch_program['hit_eps'] = hit_eps
        self.raymarch_program['t_max'] = t_max
        self.raymarch_program['max_steps'] = max_steps
        
        # Camera
        self.raymarch_program['camera_pos'] = tuple(camera_pos.astype(np.float32))
        self.raymarch_program['camera_right'] = tuple(camera_right.astype(np.float32))
        self.raymarch_program['camera_up'] = tuple(camera_up.astype(np.float32))
        self.raymarch_program['camera_forward'] = tuple(camera_forward.astype(np.float32))
        
        # Lighting
        self.raymarch_program['light_dir'] = tuple(light_dir.astype(np.float32))
        self.raymarch_program['ambient'] = ambient
        self.raymarch_program['diffuse_strength'] = diffuse_strength
        self.raymarch_program['specular_strength'] = specular_strength
        self.raymarch_program['shininess'] = shininess
        
        # Colors
        self.raymarch_program['inkColor'] = tuple(ink_color.astype(np.float32))
        self.raymarch_program['torusColor'] = tuple(torus_color.astype(np.float32))
        self.raymarch_program['backgroundColor'] = tuple(background_color.astype(np.float32))
    
    def render(self) -> np.ndarray:
        """
        Execute two-pass rendering and return RGBA image.
        
        Pass A: Build ink SDF texture in UV space
        Pass B: Ray march torus and sample ink texture
        
        Returns:
            RGBA image as numpy array [height, width, 4], float32 in [0, 1]
        """
        # ===== PASS A: Build ink SDF texture =====
        
        # Bind buffers for Pass A
        self.node_uv_buffer.bind_to_storage_buffer(1)
        self.node_radii_buffer.bind_to_storage_buffer(2)
        self.edge_uv_buffer.bind_to_storage_buffer(3)
        self.edge_radii_buffer.bind_to_storage_buffer(4)
        
        # Bind ink SDF texture as output
        self.ink_sdf_texture.bind_to_image(0, read=False, write=True)
        
        # Dispatch Pass A
        groups_x = (self.uv_texture_size + 7) // 8
        groups_y = (self.uv_texture_size + 7) // 8
        self.ink_sdf_program.run(groups_x, groups_y, 1)
        
        # Ensure Pass A completes
        self.ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                moderngl.TEXTURE_FETCH_BARRIER_BIT)
        
        # ===== PASS B: Ray march torus + sample ink texture =====
        
        # Bind ink SDF texture as input (sampler)
        self.ink_sdf_texture.use(location=0)
        
        # Bind output texture
        self.output_texture.bind_to_image(0, read=False, write=True)
        
        # Dispatch Pass B
        groups_x = (self.width + 7) // 8
        groups_y = (self.height + 7) // 8
        self.raymarch_program.run(groups_x, groups_y, 1)
        
        # Ensure Pass B completes
        self.ctx.memory_barrier(moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        
        # Read back result
        self.output_texture.read_into(self.output_buffer)
        
        # Convert to numpy array and reshape
        data = np.frombuffer(self.output_buffer.read(), dtype=np.float32)
        image = data.reshape(self.height, self.width, 4)
        
        # Clamp to [0, 1]
        image = np.clip(image, 0.0, 1.0)
        
        return image
    
    def cleanup(self):
        """Release GPU resources."""
        # Release buffers
        if self.node_uv_buffer is not None:
            self.node_uv_buffer.release()
            self.node_uv_buffer = None
        
        if self.node_radii_buffer is not None:
            self.node_radii_buffer.release()
            self.node_radii_buffer = None
        
        if self.edge_uv_buffer is not None:
            self.edge_uv_buffer.release()
            self.edge_uv_buffer = None
        
        if self.edge_radii_buffer is not None:
            self.edge_radii_buffer.release()
            self.edge_radii_buffer = None
        
        if self.output_buffer is not None:
            self.output_buffer.release()
            self.output_buffer = None
        
        # Release textures
        if self.ink_sdf_texture is not None:
            self.ink_sdf_texture.release()
            self.ink_sdf_texture = None
        
        if self.output_texture is not None:
            self.output_texture.release()
            self.output_texture = None
        
        # Release context
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None


def test_uv_gpu_availability() -> Tuple[bool, Optional[str]]:
    """
    Test if UV-based GPU rendering is available on this system.
    
    Returns:
        (available, error_message): If available is False, error_message explains why
    """
    try:
        ctx = moderngl.create_standalone_context()
        vendor = ctx.info.get('GL_VENDOR', 'Unknown')
        renderer = ctx.info.get('GL_RENDERER', 'Unknown')
        version = ctx.info.get('GL_VERSION', 'Unknown')
        ctx.release()
        return True, None
    except Exception as e:
        return False, f"Failed to create OpenGL context: {e}"
