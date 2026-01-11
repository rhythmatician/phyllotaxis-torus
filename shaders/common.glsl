// Common GLSL functions for ray marching and shading

// Smooth minimum function (Inigo Quilez formula)
float smooth_min(float d1, float d2, float k) {
    float h = clamp(k - abs(d1 - d2), 0.0, k) / k;
    return min(d1, d2) - h * h * k / 6.0;
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

// Estimate normal using gradient
vec3 estimate_normal_spheres(vec3 pos, int numSpheres, float eps) {
    // This will be implemented in the compute shader where we have access to sphere data
    // Placeholder - actual implementation will use the sphere buffer
    return vec3(0.0, 0.0, 1.0);
}
