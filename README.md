# Clearcut

Clearcut removes photo backgrounds and unwanted objects on your iPad. The models run on the device, so your photos are never uploaded.

## Install it on your iPad

1. Open your Clearcut address in Safari, for example `https://YOUR-NAME.github.io/clearcut/`.
2. Tap the Share button, then **Add to Home Screen**.
3. Open Clearcut from your Home Screen. It runs full screen, like any other app.

Each model downloads once, the first time you use the feature that needs it, and then stays on your iPad:

| Feature | Model | Download |
| --- | --- | --- |
| Remove the background | BiRefNet lite | 83 MB |
| Pick or remove objects | EfficientSAM | 21 MB |
| Fill removed areas, Best | LaMa | 90 MB |
| Fill removed areas, Fast | MI-GAN | 11 MB |
| Describe as prompt | Florence-2 PromptGen v2.0 | 270 MB |

After that, Clearcut keeps working offline.

## What you can do

- **Remove the background.** Choose a photo, drop one in, or paste one.
- **Keep only some objects.** Tap **Pick objects to keep**, then tap each object. Tap the dot on an object to drop it.
- **Remove objects.** Tap **Select objects to remove**, then tap each object you want gone, or brush over smaller things like wires. Tap **Remove**. With **Gap** set to **Fill**, Clearcut fills each gap with new background that matches the photo. **Best** uses LaMa and **Fast** uses MI-GAN. With **Transparent**, the area becomes see-through instead.
- **Describe as prompt.** Tap **Write prompt** to get a text prompt that an image generator can use to recreate the image. Choose **Prompt** for a long generator-style prompt, **Detailed** for a paragraph or **Short** for one line. Edit the text if you like, then tap **Copy**. It runs Florence-2 PromptGen on the iPad and describes the image as you would save it, with your background and crop. It runs faster with the graphics chip. On the processor, expect 20 to 60 seconds.
- **Crop.** Pick a ratio such as 1:1 or 4:5, drag the box, then tap **Apply crop**.
- **Touch up.** Paint with **Erase** or **Restore**. Mask view shows removed areas in red.
- **Change the background.** Choose transparent, the photo itself, white, black, stone, any colour or a blur of the original.
- **Save.** Tap **Save PNG** or **Save JPG**, then **Save or share**. Choose **Save Image** to add it to Photos, or **Save to Files**.

## iPad, iPhone and computers

- On iPad and iPhone, the models run on the processor. Safari 26 runs out of memory with the graphics-chip version of the model runtime, so Clearcut uses the processor version there. A photo takes about 20 to 60 seconds.
- On iPhone, Clearcut works on photos at up to 6 megapixels and uses the Fast fill by default. This keeps it within the memory an iPhone gives a web app.
- On a computer, Chrome and Edge use the graphics chip, so a photo takes a few seconds. Firefox and Safari also work.

## Publish it with GitHub Pages

You can do this in your browser on a computer. You don't need Git.

1. On GitHub, select **New repository**. Name it `clearcut`, choose **Public**, then select **Create repository**.
2. On the empty repository page, select **uploading an existing file**.
3. Unzip `clearcut-ipad.zip`. Open the `clearcut` folder, select everything inside it, and drag it all onto the upload page. GitHub keeps the folders (`m`, `vendor`, `icons`, `licenses`) as they are.
4. Wait for every file to finish uploading (about 480 MB), then select **Commit changes**.
5. Open **Settings**, then **Pages**. Under **Build and deployment**, set **Source** to **Deploy from a branch**. Choose `main` and `/ (root)`, then select **Save**.
6. After a minute or two, your app is live at `https://YOUR-NAME.github.io/clearcut/`. Replace `YOUR-NAME` with your GitHub username.

Drag the files inside the folder, not the folder itself. If you drag the folder, the address gets an extra `/clearcut/` at the end.

The web uploader accepts up to 100 files at a time and 25 MB per file. This app has 70 files and the largest is 14 MB.

A private repository needs a paid GitHub plan to use Pages. A public repository works on every plan.

## Update it later

Upload the new files the same way, keeping the same folders. Files with the same name replace the old ones. Your iPad picks up the new version the next time you open Clearcut while online. It reloads once to switch over.

## Folder layout

| Path | Contents |
| --- | --- |
| `index.html` | The app |
| `worker.js` | Model runtime and image processing, off the main thread |
| `sw.js` | Offline support |
| `manifest.webmanifest`, `icons/` | Home Screen name and icons |
| `m/` | Model files, split into chunks of 14 MB or less |
| `vendor/ort/` | ONNX Runtime Web 1.22.0 |
| `describe-worker.js` | Describe as prompt, which runs Florence-2 PromptGen |
| `licenses/` | Licences for the bundled models and runtime |

## Third-party components

| Component | Licence | Changes made |
| --- | --- | --- |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) lite weights, from the [rembg](https://github.com/danielgatis/rembg) ONNX export | MIT | Deformable convolutions and the full-resolution head rewritten as equivalent operations. Weights stored as float16. |
| [EfficientSAM](https://github.com/yformer/EfficientSAM) ViT-Tiny | Apache 2.0 | Graph simplified for fixed input sizes. Weights stored as float16. |
| [MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) 512 Places2 weights, from the [IOPaint](https://github.com/Sanster/IOPaint) TorchScript release | MIT | Exported to ONNX with constants folded. Weights stored as float16. |
| [LaMa](https://github.com/advimman/lama) Big-LaMa weights, from the [IOPaint](https://github.com/Sanster/IOPaint) TorchScript release | Apache 2.0 | Fourier transforms rewritten as equivalent matrix products so they run on the GPU. Exported to ONNX at 512 × 512. Weights stored as float16. |
| [Florence-2 PromptGen v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) by MiaoshouAI, a fine-tune of Microsoft's [Florence-2 base](https://huggingface.co/microsoft/Florence-2-base) | MIT | Exported to ONNX as an encoder and a one-step decoder. Matrix weights stored as 8-bit integers with one scale per 64 values, other weights as float16. Token embeddings looked up by the app. |
| [ONNX Runtime Web](https://onnxruntime.ai) 1.22.0 | MIT | None |
| Example photo by Rachel Michetti, from scikit-image | CC0 | Re-encoded as JPEG |
| Saira Stencil One, Instrument Sans and JetBrains Mono, from Google Fonts | SIL Open Font Licence | Loaded online, cached for offline use |

The licence texts are in the `licenses` folder.
