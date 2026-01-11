"""
GPU renderer using OpenGL compute shaders for ray marching.
Supports Intel HD Graphics 630 and other OpenGL 4.3+ compatible GPUs.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import moderngl

class GPURenderer:
    """
    OpenGL compute shader-based ray marching renderer.
    
    This renderer uses compute shaders to perform ray marching with smooth blending
    of sphere SDFs, suitable for rendering phyllotaxis visualizations.
    """
    
    def __init__(self, width: int, height: int):
        """
        Initialize OpenGL context and compile shaders.
        
        Args:
            width: Output image width in pixels
            height: Output image height in pixels
        """
        self.width = width
        self.height = height
        
        # Create OpenGL context (standalone for headless rendering)
        try:
            self.ctx = moderngl.create_standalone_context()
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenGL context: {e}")
        
        # Get GPU info
        self.gpu_info = self._get_gpu_info()
        
        # Load and compile compute shader
        self.program = self._compile_shader()
        
        # Create buffers (will be populated later)
        self.sphere_centers_buffer = None
        self.sphere_colors_buffer = None
        self.sphere_radii_buffer = None
        self.output_buffer = None
        
        # Current scene parameters
        self.num_spheres = 0
        
    def _get_gpu_info(self) -> dict:
        """Get GPU information from OpenGL context."""
        return {
            'vendor': self.ctx.info.get('GL_VENDOR', 'Unknown'),
            'renderer': self.ctx.info.get('GL_RENDERER', 'Unknown'),
            'version': self.ctx.info.get('GL_VERSION', 'Unknown'),
            'glsl_version': self.ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown'),
        }
    
    def _compile_shader(self) -> moderngl.ComputeShader:
        """Load and compile the compute shader."""
        shader_path = Path(__file__).parent.parent / 'shaders' / 'raymarch.comp'
        
        try:
            with open(shader_path, 'r') as f:
                shader_source = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Compute shader not found at {shader_path}")
        
        try:
            program = self.ctx.compute_shader(shader_source)
        except Exception as e:
            raise RuntimeError(f"Failed to compile compute shader: {e}")
        
        return program
    
    def upload_scene(
        self,
        centers: np.ndarray,
        colors: np.ndarray,
        radii: np.ndarray,
        enable_smooth_blend: bool,
        smooth_k: float,
        blend_radius: float,
        hit_eps: float,
        t_max: float,
        max_steps: int,
        camera_pos: np.ndarray,
        camera_right: np.ndarray,
        camera_up: np.ndarray,
        camera_forward: np.ndarray,
        focal_length: float,
        light_dir: np.ndarray,
        ambient: float,
        diffuse_strength: float,
        specular_strength: float,
        shininess: float,
    ):
        """
        Upload sphere data and scene parameters to GPU.
        
        Args:
            centers: Sphere centers [N, 3]
            colors: Sphere colors [N, 4] (RGBA, 0-1 range)
            radii: Sphere radii [N]
            enable_smooth_blend: Enable smooth_min blending
            smooth_k: Smoothing factor for smooth_min
            blend_radius: Radius for color blending
            hit_eps: Ray marching hit epsilon
            t_max: Maximum ray marching distance
            max_steps: Maximum ray marching steps
            camera_pos: Camera position [3]
            camera_right: Camera right vector [3]
            camera_up: Camera up vector [3]
            camera_forward: Camera forward vector [3]
            focal_length: Camera focal length
            light_dir: Light direction [3]
            ambient: Ambient light intensity
            diffuse_strength: Diffuse light strength
            specular_strength: Specular light strength
            shininess: Specular shininess exponent
        """
        self.num_spheres = len(centers)
        
        # Ensure proper data types and shapes
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 4)
        radii = np.asarray(radii, dtype=np.float32).reshape(-1)
        
        # Compute required buffer sizes in bytes
        centers_nbytes = centers.nbytes
        colors_nbytes = colors.nbytes
        radii_nbytes = radii.nbytes
        
        # Create or update buffers, recreating them if the size has changed
        recreate_sphere_buffers = (
            self.sphere_centers_buffer is None
            or self.sphere_colors_buffer is None
            or self.sphere_radii_buffer is None
            or self.sphere_centers_buffer.size != centers_nbytes
            or self.sphere_colors_buffer.size != colors_nbytes
            or self.sphere_radii_buffer.size != radii_nbytes
        )

        if recreate_sphere_buffers:
            # Release old buffers before recreating
            if self.sphere_centers_buffer is not None:
                self.sphere_centers_buffer.release()
            if self.sphere_colors_buffer is not None:
                self.sphere_colors_buffer.release()
            if self.sphere_radii_buffer is not None:
                self.sphere_radii_buffer.release()

            self.sphere_centers_buffer = self.ctx.buffer(centers.tobytes())
            self.sphere_colors_buffer = self.ctx.buffer(colors.tobytes())
            self.sphere_radii_buffer = self.ctx.buffer(radii.tobytes())

            if self.output_buffer is None:
                # Create output buffer (depends only on image size)
                output_size = self.width * self.height * 4 * 4  # RGBA float32
                self.output_buffer = self.ctx.buffer(reserve=output_size)
        else:
            # Update existing buffers without changing their size
            self.sphere_centers_buffer.write(centers.tobytes())
            self.sphere_colors_buffer.write(colors.tobytes())
            self.sphere_radii_buffer.write(radii.tobytes())
        
        # Bind buffers to shader
        self.sphere_centers_buffer.bind_to_storage_buffer(0)
        self.sphere_colors_buffer.bind_to_storage_buffer(1)
        self.sphere_radii_buffer.bind_to_storage_buffer(2)
        self.output_buffer.bind_to_storage_buffer(3)
        
        # Set uniforms
        self.program['numSpheres'].value = int(self.num_spheres)
        self.program['enable_smooth_blend'].value = bool(enable_smooth_blend)
        self.program['smooth_k'].value = float(smooth_k)
        self.program['blend_radius'].value = float(blend_radius)
        self.program['hit_eps'].value = float(hit_eps)
        self.program['t_max'].value = float(t_max)
        self.program['max_steps'].value = int(max_steps)
        
        self.program['camera_pos'].value = tuple(camera_pos.astype(np.float32))
        self.program['camera_right'].value = tuple(camera_right.astype(np.float32))
        self.program['camera_up'].value = tuple(camera_up.astype(np.float32))
        self.program['camera_forward'].value = tuple(camera_forward.astype(np.float32))
        self.program['focal_length'].value = float(focal_length)
        
        self.program['width'].value = int(self.width)
        self.program['height'].value = int(self.height)
        
        self.program['light_dir'].value = tuple(light_dir.astype(np.float32))
        self.program['ambient'].value = float(ambient)
        self.program['diffuse_strength'].value = float(diffuse_strength)
        self.program['specular_strength'].value = float(specular_strength)
        self.program['shininess'].value = float(shininess)
    
    def render(self) -> np.ndarray:
        """
        Execute compute shader and return RGBA image.
        
        Returns:
            RGBA image as numpy array [H, W, 4] with values in [0, 1]
        """
        if self.num_spheres == 0:
            # Return white image if no spheres
            return np.ones((self.height, self.width, 4), dtype=np.float32)
        
        # Compute work group count
        # Each work group is 8x8, so we need ceil(width/8) x ceil(height/8) groups
        groups_x = (self.width + 7) // 8
        groups_y = (self.height + 7) // 8
        
        # Run compute shader
        self.program.run(groups_x, groups_y, 1)
        
        # Read back results
        data = self.output_buffer.read()
        
        # Convert to numpy array and reshape
        pixels = np.frombuffer(data, dtype=np.float32)
        image = pixels.reshape(self.height, self.width, 4)
        
        # Clamp to [0, 1] range
        image = np.clip(image, 0.0, 1.0)
        
        return image
    
    def cleanup(self):
        """Release GPU resources."""
        if self.sphere_centers_buffer is not None:
            self.sphere_centers_buffer.release()
            self.sphere_centers_buffer = None
        if self.sphere_colors_buffer is not None:
            self.sphere_colors_buffer.release()
            self.sphere_colors_buffer = None
        if self.sphere_radii_buffer is not None:
            self.sphere_radii_buffer.release()
            self.sphere_radii_buffer = None
        if self.output_buffer is not None:
            self.output_buffer.release()
            self.output_buffer = None
        if self.program is not None:
            self.program.release()
            self.program = None
        if self.ctx is not None:
            self.ctx.release()
            self.ctx = None
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup


def test_gpu_availability() -> Tuple[bool, Optional[dict]]:
    """
    Test if GPU rendering is available.
    
    Returns:
        (available, gpu_info): Tuple of availability flag and GPU info dict
    """
    try:
        ctx = moderngl.create_standalone_context()
        gpu_info = {
            'vendor': ctx.info.get('GL_VENDOR', 'Unknown'),
            'renderer': ctx.info.get('GL_RENDERER', 'Unknown'),
            'version': ctx.info.get('GL_VERSION', 'Unknown'),
            'glsl_version': ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown'),
        }
        ctx.release()
        return True, gpu_info
    except Exception as e:
        return False, None
