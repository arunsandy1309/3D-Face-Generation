# Controllable 3D Face Generation with Conditional Style Code Diffusion - [Paper](https://arxiv.org/pdf/2312.13941)
This project demonstrates the generation of **3D human face videos** by utilizing a combination of **latent space interpolation**, **diffusion models**, and advanced **3D face synthesis** using the TriPlane architecture. The process involves generating realistic 3D face images from latent vectors and then interpolating these images to create smooth video sequences.

---

## Git Cloning and Installtion

To get started with the project, follow these steps:

#### 1. Clone the Repository
```bash
git clone https://github.com/arunsandy1309/3D-Face-Generation.git
```
- Make sure you have installed Visual Studio 2019 version [Build Tools](https://download.visualstudio.microsoft.com/download/pr/688bfe72-1be2-4765-bba6-06f7db68961c/1b7f19e0849eb15541f8738caab0164b8de0120da323a68643dffccfe7444d83/vs_BuildTools.exe)
- Install Nvidia CUDA 11.3 from [here](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)

#### 2. Install Python Libraries
```bash
conda create -n texface python=3.8 -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip -r install requirements.txt
```

#### 3. Download Pretrained Models and Test Dataset
1. Download [EG3D](https://drive.google.com/file/d/1ZAJxfEFbOypRMyCCRA4LfTeIbkOnZi_m/view) and [ir_se50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view), put them into 'inversion/pretrained'
2. Download [Re-PoI](https://drive.google.com/file/d/1RAJ5-B-92Ygg5aR-T1cx0U06uTsSyBqo/view).
3. Download [SCD](https://drive.google.com/file/d/10lhpORq2g_K7WVafxdRt0CVBiOQxoq90/view) and put it into 'generation/checkpoints'.
4. Download [Test Data](https://drive.google.com/file/d/1bF1fIUOZjK4JRDQKn1qhvIMU_MWxpwa_/view)

## Test Inversion
```bash
cd inversion
python scripts/test_celeba.py ./checkpoints.pt
```

## Test Generation
```bash
cd generation
python scripts/infer.py 
python scripts/gen_videos.py
```
After running the `infer.py` script, an image `output.png` is generated based on a default caption. For instance, with the default caption `"This woman has no bangs."`, the output looks like the image below:
![output](https://github.com/user-attachments/assets/44de3276-54a9-45a6-a744-95424b5be387)

If you want to customize the caption and generate an image with different descriptions, you can use the `--captions` parameter while running the script. For example:
```bash
python infer.py --captions "This man looks serious with no smile and has no fringe, and some french beard"
```
This command will generate a new image based on the provided caption. Below is an example generated using this command:
![man_with_beard](https://github.com/user-attachments/assets/b2309c7b-9d8f-4449-a5a4-e9ba429be6ce)

Once the `output.png` and `output.pt` files are created in the output directory, the next step is to run the `gen_videos.py` script. This will generate a 3D video based on the latent vectors saved in the output.pt file. The video file is saved in the output directory, and you can view the dynamic 3D face with the variations captured during interpolation.

https://github.com/user-attachments/assets/892b8206-70d6-4fdc-9760-18e4b19b0495

## Project Workflow

The workflow of this project consists of two main stages:

1. **Latent Vector Generation (`infer.py`)**: 
   - This stage generates latent vectors that encode the key features of a human face. 
   - These latent vectors are saved for further processing in the video generation stage.
   - The process uses a **diffusion model** to generate latent vectors, which are later passed to the 3D synthesis model (TriPlane).

2. **Video Generation (`gen_videos.py`)**: 
   - This stage takes the previously generated latent vectors and interpolates between them to create a series of 3D face images.
   - The interpolation process ensures smooth transitions between different expressions or angles, creating dynamic facial movements.
   - The 3D face images are synthesized using **TriPlane** architecture, and the frames are then stitched together to generate a video file (e.g., `output.mp4`).

The generated 3D faces include dynamic facial expressions and camera movements, creating a realistic 3D video.

### Step 1: `Running infer.py`
The `infer.py` script is responsible for generating a 3D face image, starting with loading the model and diffusion process, and ending with saving the generated output.

- #### 1.1 Loading Required Modules
  The script imports various libraries and functions that are necessary for the workflow:
   - PyTorch is used for deep learning and tensor operations.
   - TorchVision is used for handling image-related operations.
   - NumPy for numerical operations.
   - Custom modules like scd, TriPlaneGenerator, and GaussianDiffusion1D handle specific tasks in the 3D face generation process.
     
- #### 1.2 Setting a Random Seed
  The function set_random_seed() ensures reproducibility by setting a fixed seed across different libraries like numpy, random, and torch.
    ```python
    
    def set_random_seed(seed=None):
        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    ```
  This step ensures that every time you run the code with the same seed, you get the same output, which is crucial for experiments and debugging.

- #### 1.3 Loading the EG3D Model
  In the function load_eg3d(), the code loads the 3D face generation model TriPlaneGenerator, which has been trained to generate high-quality 3D human faces.
    ```python
    def load_eg3d(path, device='cuda'):
        eg3d = TriPlaneGenerator(...).eval().requires_grad_(False).to(device)
        ckpt = torch.load(path, map_location='cpu')
        eg3d.load_state_dict(ckpt['G_ema'], strict=False)
        eg3d.neural_rendering_resolution = 128
        return eg3d
    ```
   - TriPlaneGenerator: This is the core model responsible for generating 3D human faces.
   - G_ema: The Exponential Moving Average (EMA) version of the generator’s weights is used, which is typically more stable and produces better results than raw weights.
   - Neural rendering resolution is set to 128 to balance quality and performance.

- #### 1.4 Initializing the Diffusion Model
  A diffusion model, GaussianDiffusion1D, is initialized. This model works by refining noisy data into meaningful data (in this case, face embeddings) over multiple iterations.
    ```python
    diffusion = GaussianDiffusion1D(
            net, 
            seq_length=14, 
            timesteps=1000, 
            objective='pred_v', 
            sampling_timesteps=50
        ).to(device)
    ```
   - **Timesteps (1000):** This refers to the number of iterations during which the model refines the noisy latent vectors.
   - **Sampling Timesteps (50):** The diffusion process runs for 50 steps during the inference stage, which strikes a balance between speed and quality.
 
- #### 1.5 Camera Setup and Pose Sampling
  A virtual camera is configured to capture the face from a specific angle using the LookAtPoseSampler. This sampler helps define where the camera is looking and how it captures the 3D face.
  ```python
  camera_lookat_point = torch.tensor(eg3d.rendering_kwargs['avg_camera_pivot'], device=device)
  cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=eg3d.rendering_kwargs['avg_camera_radius'], device=device)
  ```
   - **LookAtPoseSampler:** Samples a camera pose to render the 3D face. The camera is positioned to view the generated face from the front.
   - **Focal Length:** The camera's intrinsic parameters are also set, ensuring that the camera captures the face correctly.
 
- #### 1.6 Conditional Input for Face Generation
  The model can condition its generation on various inputs, such as captions or expressions.
   - **Captions:** If text captions are provided, the model will condition the generation process on those captions.
   - **Expression Codes:** If facial expression embeddings are provided (through an .npy file), the model conditions the generation on these codes.
    ```python
    if cap is None:
        w = diffusion.sample(img=None, batch_size=test_opts.test_batch_size, cond=None, cond_scale=test_opts.cond_scale, pos=None, exp=exp)
    else:
        w = diffusion.sample(img=None, batch_size=test_opts.test_batch_size, cond=[cap] * test_opts.t
    ```

- #### 1.7 Image Synthesis and Saving
  After the diffusion process has generated the latent representations (w), the model uses the TriPlaneGenerator to synthesize the final 3D face image.
  ```python
  x = eg3d.synthesis(ws=w, c=c, noise_mode='const')['image']
  torchvision.utils.save_image(vis_img.detach().cpu(), os.path.join(test_opts.output_dir, 'output.png'), normalize=True, scale_each=True, range=(-1, 1), nrow=4)
  ```
   - **Image Output:** The synthesized face is saved as output.png.
   - **Latent Output:** The latent vector w is saved as output.pt for use in the next step (generating a video).

### Step 2: `Running gen_videos.py`
The `gen_videos.py` script is designed to produce a 3D face video by taking the latent embeddings generated by infer.py and interpolating between them to create a smooth video. Here's how the process works, step by step, followed by an in-depth look at the functions and their interactions.

- #### 2.1 Initialization and Imports
  The script first imports necessary modules like torch, imageio, os, and custom components like the 3D face generation model, sampling utilities, and latent interpolation functions.
    ```python
    import torch
    import numpy as np
    import imageio
    import os
    from model import scd
    from eg3d.training.triplane import TriPlaneGenerator
    from eg3d.camera_utils import LookAtPoseSampler
    ```
- #### 2.2 Loading Pre-trained Models and Latent Vectors
    One of the first tasks in `gen_videos.py` is loading the saved latent vector (output.pt) generated by infer.py. These latent vectors represent the internal features of the generated face.
    ```python
    latent = torch.load(latent_path)
    ```
   - The latent vectors are later interpolated to create smooth transitions, which are used to generate the frames of the video.

---

