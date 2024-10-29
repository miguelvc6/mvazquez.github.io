# Torch Tracer - GPU accelerated ray tracing with PyTorch

When thinking about tensors and computer science, two use cases instantly come to mind: computer graphics and neural networks.

If you study or work on deep learning, at some point you hear about GPUs and how they are used to accelerate parallel processes. They tell you that they are faster than CPUs for performing matrix multiplications, and that they are fundamental for neural networks.

If you read a little more about GPUs you always find the same history: a long time ago graphics were computed pixel by pixel, and it was slow. People wanted better looking video games, so the Graphics Processing Units (GPU) was invented. That it processes all pixels in the screen at the same time (actually they often work in batches), and that is good, because you can now play Skyrim at 60 fps.

In a way deep learning exists today thanks to people who wanted to play video games with better definition and faster refresh rates, because that market has been the main driver of investment and technological progress in GPUs and other parallel hardware accelerators until recently. It wasn't until 2012 with the AlexNet moment that companies started pouring money into better AI hardware.

I often work with GPUs and tensors, but always for deep learning and scientific computing. Recently I was implementing some form of self-attention transformer and I was having problems with processing the multi-attention heads in parallel for the training batches. I blame this in the deep learning frameworks, they have improved so much in the last few years that it makes it hard to justify writting custom implementations of neural networks. Why would I, instead of just picking the most recommended one? It is already fully implemented and tested.

But then some rare use case or a strange bug forces you to go deep into the architecture and see how the tensors work, or maybe your net is not training and it has to do with the size of your embedding matrix, and that's when you realize that you should have been more serious about working with your framework. That's why I wanted to spend some time making tensor calculations.

I don't know anything about computer graphics. That's why I decided to implement the **raytracing algorithm highly optimized through parallelization with pyTorch and GPU acceleration**. It allows for realistic rendering of 3D scenes with shadows, reflections, and refractions, showcasing a variety of 3D shapes and materials. A perfect example of the power of parallel computing. 🔥💨 Let your GPU go brrr and get hot.🔥💨

![Torch Tracer Example](https://github.com/miguelvc6/torch-tracer/blob/main/random_spheres.png?raw=true)
