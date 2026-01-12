"""
SDF-based GPU renderer using OpenGL compute shaders for ray marching.
Uses proper SDF primitives (capsules, torus shell) instead of sampled spheres.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import moderngl


class SDFGPURenderer:
    """
    OpenGL compute shader-based ray marching renderer using SDF primitives.

    Renders:
    - Node spheres at phyllotaxis points
    - Edge capsules as ink lines on torus inside surface
    - Smooth junctions at node-edge connections
    - Torus shell as background
    """

    def __init__(self, width: int, height: int):
        """
        Initialize OpenGL context and compile SDF shader.

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
        self.node_positions_buffer = None
        self.node_colors_buffer = None
        self.node_radii_buffer = None  # Per-node radii
        self.edge_indices_buffer = None
        self.edge_colors_buffer = None
        self.output_buffer = None

        # Current scene parameters
        self.num_nodes = 0
        self.num_edges = 0

    def _get_gpu_info(self) -> dict:
        """Get GPU information from OpenGL context."""
        return {
            "vendor": self.ctx.info.get("GL_VENDOR", "Unknown"),
            "renderer": self.ctx.info.get("GL_RENDERER", "Unknown"),
            "version": self.ctx.info.get("GL_VERSION", "Unknown"),
            "glsl_version": self.ctx.info.get("GL_SHADING_LANGUAGE_VERSION", "Unknown"),
        }

    def _compile_shader(self) -> moderngl.ComputeShader:
        """Load and compile the SDF compute shader."""
        shader_path = Path(__file__).parent.parent / "shaders" / "raymarch_sdf.comp"

        try:
            with open(shader_path, "r") as f:
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
        node_positions: np.ndarray,  # [N, 3] - 3D positions on torus
        node_colors: np.ndarray,  # [N, 4] - RGBA colors
        node_radii: np.ndarray,  # [N] - Per-node radii (world space)
        edge_indices: np.ndarray,  # [E, 2] - pairs of node indices
        edge_colors: np.ndarray,  # [E, 4] - RGBA colors for edges
        torus_R: float,  # Major radius
        torus_r: float,  # Minor radius
        shell_thickness: float,  # Thickness of inside shell band
        edge_radius: float,  # Radius of edge capsules
        smooth_k: float,  # Smoothing at junctions
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
        Upload scene data and parameters to GPU.

        Args:
            node_positions: Node 3D positions [N, 3]
            node_colors: Node RGBA colors [N, 4]
            node_radii: Per-node radii [N] in world space (calculated based on distance from camera)
            edge_indices: Edge endpoint indices [E, 2]
            edge_colors: Edge RGBA colors [E, 4]
            torus_R: Major radius of torus
            torus_r: Minor radius of torus
            shell_thickness: Thickness of inside shell band for ink
            edge_radius: Radius of edge capsule tubes
            smooth_k: Smoothing factor at junctions
            ... (camera and lighting parameters)
        """
        self.num_nodes = len(node_positions)
        self.num_edges = len(edge_indices)

        # Ensure proper data types and shapes
        node_positions = np.asarray(node_positions, dtype=np.float32).reshape(-1, 3)
        node_colors = np.asarray(node_colors, dtype=np.float32).reshape(-1, 4)
        node_radii = np.asarray(node_radii, dtype=np.float32).reshape(-1)
        edge_indices = np.asarray(edge_indices, dtype=np.int32).reshape(-1, 2)
        edge_colors = np.asarray(edge_colors, dtype=np.float32).reshape(-1, 4)

        # Compute required buffer sizes
        node_pos_nbytes = node_positions.nbytes
        node_col_nbytes = node_colors.nbytes
        node_rad_nbytes = node_radii.nbytes
        edge_idx_nbytes = edge_indices.nbytes
        edge_col_nbytes = edge_colors.nbytes

        # Create or update buffers
        recreate_buffers = (
            self.node_positions_buffer is None
            or self.node_colors_buffer is None
            or self.node_radii_buffer is None
            or self.edge_indices_buffer is None
            or self.edge_colors_buffer is None
            or self.node_positions_buffer.size != node_pos_nbytes
            or self.node_colors_buffer.size != node_col_nbytes
            or self.node_radii_buffer.size != node_rad_nbytes
            or self.edge_indices_buffer.size != edge_idx_nbytes
            or self.edge_colors_buffer.size != edge_col_nbytes
        )

        if recreate_buffers:
            # Release old buffers
            if self.node_positions_buffer is not None:
                self.node_positions_buffer.release()
            if self.node_colors_buffer is not None:
                self.node_colors_buffer.release()
            if self.node_radii_buffer is not None:
                self.node_radii_buffer.release()
            if self.edge_indices_buffer is not None:
                self.edge_indices_buffer.release()
            if self.edge_colors_buffer is not None:
                self.edge_colors_buffer.release()

            # Create new buffers
            self.node_positions_buffer = self.ctx.buffer(node_positions.tobytes())
            self.node_colors_buffer = self.ctx.buffer(node_colors.tobytes())
            self.node_radii_buffer = self.ctx.buffer(node_radii.tobytes())

            # Handle empty edge arrays
            if len(edge_indices) > 0:
                self.edge_indices_buffer = self.ctx.buffer(edge_indices.tobytes())
                self.edge_colors_buffer = self.ctx.buffer(edge_colors.tobytes())
            else:
                # Create minimal dummy buffers for empty edges
                dummy_edge = np.array([[0, 0]], dtype=np.int32)
                dummy_color = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
                self.edge_indices_buffer = self.ctx.buffer(dummy_edge.tobytes())
                self.edge_colors_buffer = self.ctx.buffer(dummy_color.tobytes())

            if self.output_buffer is None:
                # Create output buffer
                output_size = self.width * self.height * 4 * 4  # RGBA float32
                self.output_buffer = self.ctx.buffer(reserve=output_size)
        else:
            # Update existing buffers
            self.node_positions_buffer.write(node_positions.tobytes())
            self.node_colors_buffer.write(node_colors.tobytes())
            self.node_radii_buffer.write(node_radii.tobytes())
            self.edge_indices_buffer.write(edge_indices.tobytes())
            self.edge_colors_buffer.write(edge_colors.tobytes())

        # Bind buffers to shader
        self.node_positions_buffer.bind_to_storage_buffer(0)
        self.node_colors_buffer.bind_to_storage_buffer(1)
        self.node_radii_buffer.bind_to_storage_buffer(5)  # New binding for node radii
        self.edge_indices_buffer.bind_to_storage_buffer(2)
        self.edge_colors_buffer.bind_to_storage_buffer(3)
        self.output_buffer.bind_to_storage_buffer(4)

        # Set uniforms
        self.program["numNodes"].value = int(self.num_nodes)
        self.program["numEdges"].value = int(self.num_edges)
        self.program["torus_R"].value = float(torus_R)
        self.program["torus_r"].value = float(torus_r)
        self.program["shell_thickness"].value = float(shell_thickness)
        # node_radius is now per-node in buffer binding 5, not a uniform
        self.program["edge_radius"].value = float(edge_radius)
        self.program["smooth_k"].value = float(smooth_k)
        self.program["hit_eps"].value = float(hit_eps)
        self.program["t_max"].value = float(t_max)
        self.program["max_steps"].value = int(max_steps)

        self.program["camera_pos"].value = tuple(camera_pos.astype(np.float32))
        self.program["camera_right"].value = tuple(camera_right.astype(np.float32))
        self.program["camera_up"].value = tuple(camera_up.astype(np.float32))
        self.program["camera_forward"].value = tuple(camera_forward.astype(np.float32))
        self.program["focal_length"].value = float(focal_length)

        self.program["width"].value = int(self.width)
        self.program["height"].value = int(self.height)

        self.program["light_dir"].value = tuple(light_dir.astype(np.float32))
        self.program["ambient"].value = float(ambient)
        self.program["diffuse_strength"].value = float(diffuse_strength)
        self.program["specular_strength"].value = float(specular_strength)
        self.program["shininess"].value = float(shininess)

    def render(self) -> np.ndarray:
        """
        Execute compute shader and return RGBA image.

        Returns:
            RGBA image as numpy array [H, W, 4] with values in [0, 1]
        """
        if self.num_nodes == 0:
            # Return white image if no nodes
            return np.ones((self.height, self.width, 4), dtype=np.float32)

        # Compute work group count
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
        if self.node_positions_buffer is not None:
            self.node_positions_buffer.release()
            self.node_positions_buffer = None
        if self.node_colors_buffer is not None:
            self.node_colors_buffer.release()
            self.node_colors_buffer = None
        if self.node_radii_buffer is not None:
            self.node_radii_buffer.release()
            self.node_radii_buffer = None
        if self.edge_indices_buffer is not None:
            self.edge_indices_buffer.release()
            self.edge_indices_buffer = None
        if self.edge_colors_buffer is not None:
            self.edge_colors_buffer.release()
            self.edge_colors_buffer = None
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


def test_sdf_gpu_availability() -> Tuple[bool, Optional[dict]]:
    """
    Test if GPU rendering is available.

    Returns:
        (available, gpu_info): Tuple of availability flag and GPU info dict
    """
    try:
        ctx = moderngl.create_standalone_context()
        gpu_info = {
            "vendor": ctx.info.get("GL_VENDOR", "Unknown"),
            "renderer": ctx.info.get("GL_RENDERER", "Unknown"),
            "version": ctx.info.get("GL_VERSION", "Unknown"),
            "glsl_version": ctx.info.get("GL_SHADING_LANGUAGE_VERSION", "Unknown"),
        }
        ctx.release()
        return True, gpu_info
    except Exception as e:
        return False, None
