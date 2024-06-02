from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import tempfile
import zipfile
import io
import argparse
import logging
import numpy as np
from einops import rearrange, repeat
from PIL import Image
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
import os
import rembg
from torchvision.transforms import v2
import trimesh
import requests

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    device0 = torch.device('cuda:0')
    device1 = torch.device('cuda:1')
else:
    device0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = device0

# Define the cache directory for model files
model_cache_dir = './models/'
os.makedirs(model_cache_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

logging.getLogger("httpx").setLevel(logging.WARNING)

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        client_address = self.client_address[0]
        logging.info(f"Received request from {client_address}")

        if self.headers["Authorization"] != args.auth_token:
            logging.warning(f"`Unauthorized access attempt` from {client_address}")
            self.send_response(401)
            self.end_headers()
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST"}
        )

        processed_images = None

        if "image" in form:
            image_file = form["image"].file

            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(image_file.read())
                temp_file_path = temp_file.name
            
            image_file = temp_file_path
        elif "multi_image_urls" in form:
            multi_image_urls = form["multi_image_urls"].value
            urls = multi_image_urls.split(',')
            processed_images = []

            for image_url in urls:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image_file = io.BytesIO(response.content)

                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(image_file.read())
                        temp_file_path = temp_file.name

                    image = Image.open(temp_file_path).convert('RGB')
                    image_np = np.asarray(image, dtype=np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()

                    processed_images.append(image_tensor)
                else:
                    logging.warning(f"Failed to download image from {image_url}")
                    self.send_response(400)
                    self.end_headers()
                    return

            processed_images = torch.stack(processed_images)

        elif "image_url" in form:
            image_url = form["image_url"].value
            # Download the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                image_file = io.BytesIO(response.content)

                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(image_file.read())
                    temp_file_path = temp_file.name

                image_file = temp_file_path

            else:
                logging.warning(f"Failed to download image from {image_url}")
                self.send_response(400)
                self.end_headers()
                return
        else:
            logging.warning("No image or from_url field found in the request")
            self.send_response(400)
            self.end_headers()
            return

        if processed_images is None:

            ###############################################################################
            # Stage 1: Multiview generation.
            ###############################################################################

            rembg_session = None if args.no_rembg else rembg.new_session()

            name = os.path.basename(image_file).split('.')[0]
            print(f'Imagining {name} ...')

            # remove background optionally
            input_image = Image.open(image_file)
            if not args.no_rembg:
                input_image = remove_background(input_image, rembg_session)
                input_image = resize_foreground(input_image, 0.85)
            
            # sampling
            output_image = pipeline(
                input_image, 
                num_inference_steps=args.diffusion_steps, 
            ).images[0]

            output_image.save(os.path.join(image_path, f'{name}.png'))
            print(f"Image saved to {os.path.join(image_path, f'{name}.png')}")

            images = np.asarray(output_image, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
            images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
        else:
            images = rearrange(processed_images, '(b n m) c h w -> (n m b) c h w', n=3, m=2)  # Rearrange to (6, 3, 320, 320)
            name = "multi-images"

        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device1)
        chunk_size = 20 if IS_FLEXICUBES else 1

        print(f'Creating {name} ...')

        images = images.unsqueeze(0).to(device1)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        with torch.no_grad():
            # get triplane
            planes = model.forward_planes(images, input_cameras)

            # get mesh
            mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

            mesh_out = model.extract_mesh(
                planes,
                use_texture_map=True,
                **infer_config,
            )

            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
            print(f"Mesh saved to {mesh_path_idx}")

        mesh = trimesh.load(mesh_path_idx)

        rotation_x = np.radians(-90)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=rotation_x,
            direction=[1, 0, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)

        rotation_y = np.radians(-45)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=rotation_y,
            direction=[0, 1, 0],
            point=[0, 0, 0]
        )
        mesh.apply_transform(rotation_matrix)

        glb_path = os.path.join(mesh_path, f'{name}.glb')
        mesh.export(glb_path)

        logging.info(f"Sending response to {client_address}")
        self.send_response(200)
        self.send_header("Content-Type", "model/gltf-binary")
        self.send_header("Content-Disposition", "attachment; filename=result.glb")
        self.end_headers()

        with open(glb_path, "rb") as glb_file:
            self.wfile.write(glb_file.read())

        logging.info(f"Request from {client_address} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP server with authentication")
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument("--auth-token", required=True, help="Authentication token")
    parser.add_argument("--port", type=int, default=8112, help="Port number (default: 8112)")
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
    parser.add_argument('--diffusion_steps', type=int, default=75, help='Denoising Sampling steps.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config_name = os.path.basename(args.config).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False

    # load diffusion model
    print('Loading diffusion model ...')
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2", 
        custom_pipeline="zero123plus",
        torch_dtype=torch.float16,
        cache_dir=model_cache_dir
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )

    # load custom white-background UNet
    print('Loading custom white-background unet ...')
    if os.path.exists(infer_config.unet_path):
        unet_ckpt_path = infer_config.unet_path
    else:
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model", cache_dir=model_cache_dir)
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(device0)

    # load reconstruction model
    print('Loading reconstruction model ...')
    model = instantiate_from_config(model_config)
    if os.path.exists(infer_config.model_path):
        model_ckpt_path = infer_config.model_path
    else:
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model", cache_dir=model_cache_dir)
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device1)
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device1, fovy=30.0)
    model = model.eval()

    # make output directories
    image_path = os.path.join(args.output_path, config_name, 'images')
    mesh_path = os.path.join(args.output_path, config_name, 'meshes')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)

    server_address = ("", args.port)
    httpd = HTTPServer(server_address, RequestHandler)
    logging.info(f"Server running on port {args.port}...")
    httpd.serve_forever()
