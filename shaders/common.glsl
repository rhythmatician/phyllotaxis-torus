// Common GLSL functions for ray marching and shading
//
// NOTE: This file is currently not included in any shader source.
// The functions below (smooth_min, sphere_sdf, phong_shade) are duplicated
// in raymarch.comp for direct compilation without includes.
//
// This file is preserved for potential future use if GLSL include/concatenation
// mechanisms are implemented in the build pipeline.
//
// If you wish to centralize common GLSL utilities, ensure they are pulled
// into other shaders via your project's GLSL include/concatenation mechanism.

// Smooth minimum function (Inigo Quilez polynomial formula)
float smooth_min(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Sphere SDF
float sphere_sdf(vec3 pos, vec3 center, float radius) {
    return length(pos - center) - radius;
}

// Phong shading calculation
vec3 phong_shade(
    vec3 base_color,
    vec3 normal,
    vec3 view_dir,
    vec3 light_dir,
    float ambient,
    float diffuse_strength,
    float specular_strength,
    float shininess
) {
    // Normalize inputs
    light_dir = normalize(light_dir);
    normal = normalize(normal);
    view_dir = normalize(view_dir);
    
    // Ambient component
    vec3 ambient_col = ambient * base_color;
    
    // Diffuse component
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_col = diffuse_strength * diff * base_color;
    
    // Specular component (Blinn-Phong)
    vec3 half_vec = normalize(view_dir + light_dir);
    float spec = max(dot(normal, half_vec), 0.0);
    float spec_pow = pow(spec, shininess);
    vec3 specular_col = specular_strength * spec_pow * vec3(1.0);
    
    return ambient_col + diffuse_col + specular_col;
}
