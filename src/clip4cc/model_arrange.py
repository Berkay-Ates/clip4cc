import argparse
import os

import torch
from torch.utils.data import DataLoader

from clip4cc.data_loader import Clip4CCDataLoader

from .file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from .modeling import CLIP4IDC


def assign_model_args(data_path, features_path, init_model):
    return argparse.Namespace(
        do_pretrain=False,
        do_train=False,
        do_eval=True,
        data_path="",
        features_path="",
        num_thread_reader=1,
        lr=0.0001,
        epochs=1,
        batch_size=32,
        batch_size_val=32,
        lr_decay=0.9,
        n_display=100,
        seed=42,
        max_words=77,
        feature_framerate=1,
        margin=0.1,
        hard_negative_rate=0.5,
        negative_weighting=1,
        n_pair=1,
        output_dir="output/",
        cross_model="cross-base",
        decoder_model="decoder-base",
        do_lower_case=False,
        warmup_proportion=0.1,
        gradient_accumulation_steps=1,
        cache_dir=os.path.join(
            str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed"
        ),
        fp16=False,
        fp16_opt_level="O1",
        task_type="retrieval",
        datatype="levircc",  # dataloader fixed as loading separate images
        coef_lr=1.0,
        use_mil=False,
        sampled_use_mil=False,
        text_num_hidden_layers=12,
        visual_num_hidden_layers=12,
        intra_num_hidden_layers=9,
        cross_num_hidden_layers=2,
        freeze_layer_num=0,
        linear_patch="2d",
    )


def load_model(args, device, model_file=None):
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location="cpu")

        print("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = (
            args.cache_dir
            if args.cache_dir
            else os.path.join(
                str(PYTORCH_PRETRAINED_BERT_CACHE), "distributed"
            )
        )
        model = CLIP4IDC.from_pretrained(
            args.cross_model,
            args.decoder_model,
            cache_dir=cache_dir,
            state_dict=model_state_dict,
            task_config=args,
        )

        model.to(device)
    else:
        model = None
    return model.eval()


def get_text_vec(model, text, device, dummy_img):
    dataset = Clip4CCDataLoader(
        bef_img_path=dummy_img, aft_img_path=dummy_img, text_caption=text
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sequence_output, visual_output = eval_model(
        model=model, dataloader=dataloader, device=device
    )

    return sequence_output


def get_img_pair_vec(model, img1_pth, img2_pth, device):
    dataset = Clip4CCDataLoader(
        bef_img_path=img1_pth, aft_img_path=img2_pth, text_caption=""
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sequence_output, visual_output = eval_model(
        model=model, dataloader=dataloader, device=device
    )

    return visual_output


def get_single_output(model, img1_pth, img2_pth, text, device):
    dataset = Clip4CCDataLoader(
        bef_img_path=img1_pth, aft_img_path=img2_pth, text_caption=text
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sequence_output, visual_output = eval_model(
        model=model, dataloader=dataloader, device=device
    )

    return sequence_output, visual_output


def eval_model(model, dataloader, device):
    if hasattr(model, "module"):
        model = model.module.to(device)
    else:
        model = model.to(device)

    with torch.no_grad():
        for bid, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            (
                input_ids,
                input_mask,
                segment_ids,
                bef_image,
                aft_image,
                image_mask,
            ) = batch
            image_pair = torch.cat([bef_image, aft_image], 1)

            # Modelden metin ve görüntü çıktılarını al
            sequence_output, _ = model.get_sequence_output(
                input_ids, segment_ids, input_mask
            )
            visual_output, _ = model.get_visual_output(image_pair, image_mask)

            visual_output = visual_output / visual_output.norm(
                dim=-1, keepdim=True
            )
            sequence_output = sequence_output / sequence_output.norm(
                dim=-1, keepdim=True
            )

    return sequence_output.squeeze(), visual_output.squeeze()
