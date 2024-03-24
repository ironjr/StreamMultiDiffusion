from typing import Dict, List, Tuple, Union


quality_prompt_list = [
    {
        "name": "(None)",
        "prompt": "{prompt}",
        "negative_prompt": "nsfw, lowres",
    },
    {
        "name": "Standard v3.0",
        "prompt": "{prompt}, masterpiece, best quality",
        "negative_prompt": "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
    },
    {
        "name": "Standard v3.1",
        "prompt": "{prompt}, masterpiece, best quality, very aesthetic, absurdres",
        "negative_prompt": "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
    },
    {
        "name": "Light v3.1",
        "prompt": "{prompt}, (masterpiece), best quality, very aesthetic, perfect face",
        "negative_prompt": "nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn",
    },
    {
        "name": "Heavy v3.1",
        "prompt": "{prompt}, (masterpiece), (best quality), (ultra-detailed), very aesthetic, illustration, disheveled hair, perfect composition, moist skin, intricate details",
        "negative_prompt": "nsfw, longbody, lowres, bad anatomy, bad hands, missing fingers, pubic hair, extra digit, fewer digits, cropped, worst quality, low quality, very displeasing",
    },
]

style_list = [
    {
        "name": "(None)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "{prompt}, cinematic still, emotional, harmonious, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "nsfw, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "{prompt}, cinematic photo, 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "nsfw, drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "{prompt}, anime artwork, anime style, key visual, vibrant, studio anime, highly detailed",
        "negative_prompt": "nsfw, photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "{prompt}, manga style, vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "nsfw, ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "{prompt}, concept art, digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "nsfw, photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "{prompt}, pixel-art, low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "nsfw, sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "{prompt}, ethereal fantasy concept art, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "nsfw, photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "{prompt}, neonpunk style, cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, ultra detailed, intricate, professional",
        "negative_prompt": "nsfw, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "{prompt}, professional 3d model, octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "nsfw, ugly, deformed, noisy, low poly, blurry, painting",
    },
]


_style_dict = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in style_list}
_quality_dict = {k["name"]: (k["prompt"], k["negative_prompt"]) for k in quality_prompt_list}


def preprocess_prompt(
    positive: str,
    negative: str = "",
    style_dict: Dict[str, dict] = _quality_dict,
    style_name: str = "Standard v3.1", # "Heavy v3.1"
    add_style: bool = True,
) -> Tuple[str, str]:
    p, n = style_dict.get(style_name, style_dict["(None)"])

    if add_style and positive.strip():
        formatted_positive = p.format(prompt=positive)
    else:
        formatted_positive = positive

    combined_negative = n
    if negative.strip():
        if combined_negative:
            combined_negative += ", " + negative
        else:
            combined_negative = negative

    return formatted_positive, combined_negative


def preprocess_prompts(
    positives: List[str],
    negatives: List[str] = None,
    style_dict = _style_dict,
    style_name: str = "Manga", # "(None)"
    quality_dict = _quality_dict,
    quality_name: str = "Standard v3.1", # "Heavy v3.1"
    add_style: bool = True,
    add_quality_tags = True,
) -> Tuple[List[str], List[str]]:
    if negatives is None:
        negatives = ['' for _ in positives]

    positives_ = []
    negatives_ = []
    for pos, neg in zip(positives, negatives):
        pos, neg = preprocess_prompt(pos, neg, quality_dict, quality_name, add_quality_tags)
        pos, neg = preprocess_prompt(pos, neg, style_dict, style_name, add_style)
        positives_.append(pos)
        negatives_.append(neg)
    return positives_, negatives_


def print_prompts(
    positives: Union[str, List[str]],
    negatives: Union[str, List[str]],
    has_background: bool = False,
) -> None:
    if isinstance(positives, str):
        positives = [positives]
    if isinstance(negatives, str):
        negatives = [negatives]

    for i, prompt in enumerate(positives):
        prefix = ((f'Prompt{i}' if i > 0 else 'Background Prompt')
                  if has_background else f'Prompt{i + 1}')
        print(prefix + ': ' + prompt)
    for i, prompt in enumerate(negatives):
        prefix = ((f'Negative Prompt{i}' if i > 0 else 'Background Negative Prompt')
                  if has_background else f'Negative Prompt{i + 1}')
        print(prefix + ': ' + prompt)
