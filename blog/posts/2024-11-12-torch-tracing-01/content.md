# Torch Tracer - GPU accelerated ray-tracing with PyTorch

## Torch tracer

When you think about tensors in computer science, two areas immediately stand out: computer graphics and neural networks.

In the space of deep learning, it's inevitable to encounter discussions about GPUs and their role in accelerating parallel computations. GPUs are faster than CPUs for matrix multiplications, which makes them fundamental for neural networks.

If you read a little more about GPUs you'll find the following history: a long time ago graphics were computed pixel by pixel, and it was slow. People wanted better looking video games, so they came up with Graphics Processing Units (GPU). They process pixels in parallel (often in batches), and that is good, because now video games look great.

In a way deep learning exists today thanks to people who wanted to play video games with better definition and faster refresh rates, because that market has historically been the main driver of investment and technological progress in GPUs and other parallel hardware accelerators. It wasn't until 2012 with the AlexNet moment that companies started pouring money into better AI hardware.

I often work with GPUs and tensors, but always for deep learning and scientific computing. Recently, while implementing a self-attention transformer, I encountered issues processing multiple attention heads in parallel during training. It was a fairly complex tensor operation and it was hard to get the shapes right. I blame this on the deep learning frameworks: they have improved so much in the last few years that it makes it hard to justify writing custom implementations of neural networks, and these skills get rusty. Why would I write my own, instead of just picking one of the recommended ones, which are already fully implemented and tested?

But then, when your model isn't training due to the size of your embedding matrix, or you're working with a rare use case where no implemented modules exist, or a strange bug forces you deep into the architecture, you realize the importance of understanding tensors and linear algebra. For this reason I wanted to spend some time making tensor calculations and sharpening my skills.

That's why I decided to implement the **raytracing algorithm, highly optimized through parallelization, with PyTorch and GPU acceleration**. It allows for realistic rendering of 3D scenes with shadows, reflections, and refractions, showcasing a variety of 3D shapes and materials. A perfect example of the power of parallel computing. 🔥💨 Let your GPU go brrr and get hot.🔥💨

[TorchTracer GitHub Repository](https://github.com/miguelvc6/torch-tracer)

ray-tracing is a rendering technique that simulates how light interacts with objects to produce highly realistic images. I have based my implementation on the book [&#34;ray-tracing in One Weekend&#34; by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html)[^1], in which a basic raytracer is implemented using C++ in a sequential manner, but I have implemented it in Python with PyTorch and parallelized the raytracing.

<p align="center">
  <img src="https://github.com/miguelvc6/torch-tracer/blob/main/random_spheres.png?raw=true" width="80%" />
</p>

<p style="text-align:center; font-style: italic;">Rendering Results with TorchTracer: Random Spheres.</p>

This is the same image as in the book (modulo random sphere placement) rendered using the Torch Tracer. The scene includes various 3D shapes, materials, and lighting effects to demonstrate the capabilities of the raytracer.

## Ray-Tracing Algorithm

The essence of ray-tracing is to render an image pixel by pixel by simulating the behavior of light rays. By defining an origin point and a grid of pixels in space, rays of light are cast from the origin towards each pixel, resembling the way a camera captures a scene.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Ray_trace_diagram.svg/1280px-Ray_trace_diagram.svg.png" width="80%" />
</p>
<p style="text-align:center; font-style: italic;">The Ray-Tracing Algorithm builds an image by extending rays into a scene and bouncing them off surfaces and towards sources of light to approximate the color value of pixels.
<br/>
<small>Image source: <a href="https://commons.wikimedia.org/wiki/File:Ray_trace_diagram.svg">Wikimedia Commons</a></small>
</p>

Then, the collisions of the rays with the world objects are computed. Once the intersected objects have been identified, the amount of incoming light is calculated at the points of intersection and, depending on the objects' material properties, the rays bounce (solid and reflective objects) or go through them (transparent or translucent objects).

This process is repeated for a maximum number of bounces, accumulating the light contributions from each surface interaction until the final color of each pixel is determined.

For the collision with surfaces, the following simplified code shows the parallel computation:

```python
@jaxtyped(typechecker=typechecker)
class Sphere(Hittable):
    def __init__(self, center: Float[t.Tensor, "3"], radius: float, material: Material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(
        self,
        pixel_rays: Float[t.Tensor, "N 3 2"],
        t_min: float,
        t_max: float,
    ) -> HitRecord:
        """Calculate ray-sphere intersections.

        Uses quadratic formula to solve: |P(t) - C|^2 = r^2
        where P(t) = A + tb is the ray equation
        """
        ray_origin = pixel_rays[:, :, 0]
        ray_direction = pixel_rays[:, :, 1]

        oc = ray_origin - self.center
        a = (ray_direction * ray_direction).sum(dim=-1)
        half_b = (oc * ray_direction).sum(dim=-1)
        c = (oc * oc).sum(dim=-1) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        record = HitRecord.empty(discriminant.shape)
        hit_mask = discriminant >= 0

        if not hit_mask.any():
            return record

        sqrtd = t.sqrt(discriminant[hit_mask])
        root = (-half_b[hit_mask] - sqrtd) / a[hit_mask]

        # Try second root if first is invalid
        second_root_mask = root <= t_min
        root[second_root_mask] = (
            (-half_b[hit_mask][second_root_mask] + sqrtd[second_root_mask]) /
            a[hit_mask][second_root_mask]
        )

        valid_hit = (root >= t_min) & (root <= t_max)
        hit_mask[hit_mask] &= valid_hit

        if not hit_mask.any():
            return record

        # Calculate hit points and surface normals
        record.hit = hit_mask
        record.t[hit_mask] = root[valid_hit]
        record.point[hit_mask] = (
            ray_origin[hit_mask] +
            root[valid_hit].unsqueeze(-1) *
            ray_direction[hit_mask]
        )
        outward_normal = (record.point[hit_mask] - self.center) / self.radius
        record.set_face_normal(ray_direction[hit_mask], outward_normal)

        # Set material properties
        self.material.apply_properties(record, hit_mask)

        return record
```

<p style="text-align:center; font-style: italic;">Light rays hitting on a Sphere class</p>
<br/>

This process simulates how light and colors work in a real, physical scenario, with a main difference being that rays are cast from the camera (often called the origin) towards the scene, rather than from light sources, primarily for computational efficiency, since computing the rays from the light-sources would imply computing many rays that would never reach the camera.

In the actual implementation many rays are shot through each pixel, with a pixel being a small rectangle in space, and the resulting colors are averaged. This creates a smoothing effect called antialiasing, which reduces the jagged, 'stair-step' appearance on object edges, and enhances visual quality.

This algorithm allows for significant optimizations through parallelization, and here is where tensor calculus comes in hand with PyTorch. The only necessarily sequential process is the rebound of the rays in the surfaces, and every step can be paralellized.

Let's consider a $H \times W$ viewport, with $S$ rays per pixel and $max\\_depth$ total bounces. Then, the algorithm needs to compute at most $H \cdot W \cdot S \cdot max\\_depth$ operations of hit check, ray bounce and color calculation. In practice there are less operations since the rays that do not collide with a surface go way and are ignored thereafter.

This is a lot of calculations if we want a high definition image, but all the rays can be processed in parallel for each bounce, which can speed things up by a lot. If tha GPU has enough memory to fit all the data,the entire process can be computed in $max\\_depth$ steps. If the data does not fit fully in memory, it can still be accelerated by processing in batches.

In the following code block, which is a simplification from the repository, the ray-tracing process is shown:

1. Traces rays through the scene in parallel
2. Handles ray-object intersections
3. Computes material interactions and scattered rays
4. Accumulates color contributions from multiple bounces
5. Processes background colors for rays that miss all objects

The code leverages PyTorch tensors to perform these calculations efficiently in parallel on the GPU.

```python
@jaxtyped(typechecker=typechecker)
def ray_color(
    self,
    pixel_rays: Float[t.Tensor, "N 3 2"],
    world: Hittable,
) -> Float[t.Tensor, "N 3"]:
    """Trace rays through the scene and compute colors.

    Args:
        pixel_rays: Tensor of ray origins and directions
        world: Collection of hittable objects in the scene
    """
    N = pixel_rays.shape[0]
    colors = t.zeros((N, 3), device=device)
    attenuation = t.ones((N, 3), device=device)
    rays = pixel_rays
    active_mask = t.ones(N, dtype=t.bool, device=device)

    for _ in range(self.max_depth):
        if not active_mask.any():
            break

        # Test ray intersections with scene objects
        hit_record = world.hit(rays, 0.001, float("inf"))

        # Handle rays that hit the background
        no_hit_mask = (~hit_record.hit) & active_mask
        if no_hit_mask.any():
            ray_dirs = F.normalize(rays[no_hit_mask, :, 1], dim=-1)
            t_param = 0.5 * (ray_dirs[:, 1] + 1.0)
            background_colors = (1.0 - t_param).unsqueeze(-1) * t.tensor(
                [1.0, 1.0, 1.0], device=device
            )
            background_colors += t_param.unsqueeze(-1) * t.tensor(
                [0.5, 0.7, 1.0], device=device
            )
            colors[no_hit_mask] += attenuation[no_hit_mask] * background_colors
            active_mask[no_hit_mask] = False

        # Process material interactions for rays that hit objects
        hit_mask = hit_record.hit & active_mask
        if hit_mask.any():
            hit_indices = hit_mask.nonzero(as_tuple=False).squeeze(-1)
            material_types_hit = hit_record.material_type[hit_indices]

            # Handle each material type separately
            for material_type in [
                MaterialType.Lambertian,
                MaterialType.Metal,
                MaterialType.Dielectric,
            ]:
                material_mask = material_types_hit == material_type
                if material_mask.any():
                    indices = hit_indices[material_mask]
                    # Calculate scattered rays and attenuation based on material properties
                    scatter_mask, mat_attenuation, scattered_rays = (
                        self._scatter_ray(
                            material_type, rays[indices], hit_record, indices
                        )
                    )
                    attenuation[indices] *= mat_attenuation
                    rays[indices] = scattered_rays
                    # Mark absorbed rays as inactive
                    terminated = ~scatter_mask
                    if terminated.any():
                        term_indices = indices[terminated]
                        active_mask[term_indices] = False

    return colors
```

<p style="text-align:center; font-style: italic;">Hit & bounce algorithm, simplified from the TorchTracer repository</p>
<br/>

## Comparison with the Book and Experiments

The **book** implements the raytracer in C++. For every pixel in the view plane, the book computes `samples_per_pixel` rays through the pixel and then traces the ray through the scene to compute the color of the pixel, for at most `max_depth` bounces. This is done sequentially for each pixel, sample and bounce.

The **Torch Tracer** uses parallelization with PyTorch to compute the rays in parallel for every pixel and sample in the view plane. This allows for a significant speedup in rendering time.

This means that for an image with 1920x1080 pixels and 120 samples per pixel, the book computes 1920x1080x120=248,832,000 rays to render the image. Every ray may bounce multiple times, for a maximum of `max_depth` bounces. This means that the book computes at most 248,832,000 x 50 = **12,441,600,000 rays** to render the image.

The **Torch Tracer** computes the same number of rays, but does it in parallel for every pixel and sample. This means that, if enough GPU memory is available, the Torch Tracer can render the image in just `max_depth`, so **50 passes in this case**, in parallel.

In practice, I evaluated the performance by generating the same scene as in the book's repository. The Torch Tracer takes approximately 170 seconds on my GPU (a laptop NVIDIA GeForce RTX 4050 with 8GB of memory), compared to ~645 seconds for the book's C++ sequential implementation. That's a speedup of ~3.79x. The main limiting factor for the Torch Tracer is the GPU memory, for which I have implemented a sequential batching system, but potentially every ray-tracing bounce could be done in parallel.

This demonstrates how leveraging GPU parallelization can dramatically improve performance, making it feasible to render complex scenes much faster compared to a simpler sequential method.

## Features

With the core concepts of ray tracing in mind, let's explore some of the advanced features that Torch Tracer brings to the table.

-   🚀 GPU Acceleration with PyTorch

    -   Batched ray processing for efficient GPU utilization
    -   Parallel computation of ray intersections and color calculations
    -   Configurable batch size to manage memory usage

-   🎨 Advanced Ray-Tracing Capabilities

    -   Multiple ray bounces with configurable maximum depth
    -   Anti-aliasing through multiple samples per pixel
    -   Depth of field and defocus blur effects
    -   Realistic shadows and reflections

-   ✨ Material System

    -   Lambertian (diffuse) surfaces with matte finish
    -   Metal surfaces with configurable reflectivity and fuzz
    -   Dielectric (glass) materials with refraction
    -   Support for multiple materials in a single scene

-   📷 Camera System

    -   Configurable field of view and aspect ratio
    -   Adjustable camera position and orientation
    -   Focus distance and defocus angle controls
    -   Support for different image resolutions

-   🛡️ Type Safety

    -   Static type checking with jaxtyping
    -   Runtime type validation with typeguard
    -   Array shape and dtype validation

## Materials

<p align="center">
  <img src="https://github.com/miguelvc6/torch-tracer/blob/main/image_material_showcase.png?raw=true" width="80%" />
</p>
<p style="text-align:center; font-style: italic;">A showcase of the three material implementations. </p>
<br/>

### Lambertian: Diffuse material with matte finish

Lambertian materials simulate diffuse surfaces that scatter light in random directions. When a ray hits a Lambertian surface, it bounces in a random direction within the hemisphere centered around the surface normal. This creates the characteristic matte appearance we see in objects like chalk or unfinished wood, where light seems to spread evenly in all directions.

### Metal: Reflective material with configurable fuzz for glossiness

Metal surfaces are all about reflection. When a ray hits a metallic surface, it bounces following the law of reflection: the angle of incidence equals the angle of reflection. To create more realistic metals that aren't perfectly mirror-like, I add a "fuzz" parameter that randomly perturbs the reflected ray. Higher fuzz values create a more brushed or tarnished metal look.

### Dielectric: Glass-like material with refraction (configurable index)

Dielectric materials like glass or water handle both reflection and refraction. When a ray hits a dielectric surface, it splits into a reflected and a refracted component based on Snell's law and the material's refractive index. The ratio between reflection and refraction varies with the angle of incidence, creating effects like total internal reflection when light tries to exit the material at shallow angles.

## Following Steps

I really like to implement this type of projects that involves cool renderings and simulations. I feel like I have improved quite a bit my tensor manipulation skills, and I have managed to succesfully work on computer graphics.

There are many additional things I could implement, and the authors of Ray-tracing in One Weekend have written two more books. I plan to increase the scope of my Torch Tracer, but in a more *AI engineer* way. I have recently been working with LLM agents, so my idea is to write one that is able to implement the second book of the series in PyTorch taking my current Torch Tracer as starting point. I expect to upload a couple of blog posts about agents, and I will try to make this work.

## References

[^1]: “Ray Tracing in One Weekend.” [raytracing.github.io/books/RayTracingInOneWeekend.html](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
