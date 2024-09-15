from diffusers import StableDiffusionImg2ImgPipeline
import torch
import torchvision.transforms as transforms
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os

def plt_show_image(image):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def seed_everything(seed=42):
    import torch
    import numpy as np
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_image(image, filename):
    img = Image.fromarray((image * 255).astype('uint8'))
    img.save(filename)


@torch.no_grad()
def preprocess(image):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize(512),  # Resize to the same size as your model input
        transforms.ToTensor(),  # Convert PIL image to Torch tensor
    ])
    image_tensor = preprocess_pipeline(image)
    image_tensor.unsqueeze_(0)  # Add batch dimension
    return image_tensor


@torch.no_grad()
def generate_img2img_simplified(
        prompt,
        init_image,
        negative_prompt=[""],
        strength=0.8,  # strength of the image conditioning
        batch_size=1,
        num_inference_steps=50,
        do_classifier_free_guidance=True,
        guidance_scale=7.5
):
    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)

    # get prompt text embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]

    # get unconditional embeddings for classifier free guidance
    uncond_tokens = negative_prompt
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

    # classifier free guidance
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    # 以上内容与txt2img对txt的处理方法完全一致
    #####################################################################
    latents_dtype = text_embeddings.dtype
    if isinstance(init_image, PIL.Image.Image):
        init_image = preprocess(init_image)
    # 这里的预处理就是 将宽resize到512，然后转成tensor,   #移到GPU上
    init_image = init_image.tobytes(device=pipe.device, latents_dtype=latents_dtype)

    # 丢进VAE中,得到latent distribution      也就是原图经过处理得到的(小)图
    init_latent_dist = pipe.vae.encode(init_image).latent_dist
    init_latents = init_latent_dist.sample()
    init_latents = 0.18215 * init_latents

    # 获取init_timesteps
    offset = pipe.scheduler.config.get('steps_offset', 0)  # 一般是1
    init_timesteps = int(num_inference_steps * strength) + offset  # 假设当前是40+1=41
    init_timesteps = min(init_timesteps, num_inference_steps)

    # 寻找这41步在1000这个尺度上的映射
    init_timesteps_1000 = pipe.scheduler.timesteps[-init_timesteps]  # 801
    # 如果有多个batch_size
    init_timesteps_1000 = torch.tensor([init_timesteps_1000] * batch_size).to(device=pipe.device, dtype=latents_dtype)

    # 此时已经知道所有的噪声水平对应的索引,,开始加噪
    # random_normal 均值0，方差1的噪声
    noise = torch.randn(init_latents.shape, device=pipe.device, dtype=latents_dtype)
    init_latents = pipe.scheduler.add_noise(init_latents, noise, init_timesteps_1000)

    #到此为止,,加噪结束,,因此将init_latents作为latents开始进行Sampling
    latents=init_latents

    t_start = max(num_inference_steps-init_timesteps+offset,0)  #50-41+1=10
    final_num_inference_steps = pipe.scheduler.timesteps[t_start:].to(device=pipe.device)


    #####################################################################

    for i, t in enumerate(pipe.progress_bar(final_num_inference_steps)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


if __name__ == "__main__":  # vanilla
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)



    seed_everything(1022)


    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "/data_disk/dyy/Ghibli-Diffusion",
        revision="fp16", torch_dtype=torch.float16,
    ).to(device)

    # Load the initial image if required.
    init_image = Image.open("./jpgs/g2.jpg")
    # Now we can use the function to generate an image.

    image = generate_img2img_simplified(
        prompt=["ghibli style, A cute girl in a tank top "],
        negative_prompt=[""],
        init_image=init_image,
        strength=0.75,
        batch_size=1,
    )

    # plt_show_image(image[0])
    save_image(image[0], "./result.png")
