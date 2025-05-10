from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

# def get_vit_attention_map(model, image_tensor, device='cuda'):
#     """
#     Extracts the attention map from the last self-attention layer of a ViT model.
#     Returns the average attention from [CLS] token to all image patches.
#     """
#     model.eval()
#     with torch.no_grad():
#         inputs = image_tensor.to(device)
#         outputs = model.encode_image(inputs)

#         # Get attention weights
#         if hasattr(model.visual, 'attnpool'):
#             raise NotImplementedError("Attention pooling not supported in this function.")

#         attn_blocks = model.visual.transformer.resblocks
#         last_attn = attn_blocks[-1].attn.get_attention_map()
        
#         # Use attention from the CLS token to all patches
#         cls_attn = last_attn[:, 0, 1:]  # (batch, patches)
#         attn_map = cls_attn.mean(dim=0)  # Average over heads
#         return attn_map.cpu().numpy().reshape(1, -1)


# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision.transforms.functional as F
# from PIL import Image

# def show_attention_overlay(image_tensor, attention_map, title='Attention'):
#     """
#     Overlays attention heatmap on the input image.
#     image_tensor: (1, 3, H, W) torch.Tensor
#     attention_map: (1, num_patches) numpy array
#     """
#     # Convert image to PIL
#     image = F.to_pil_image(image_tensor.squeeze(0).cpu())
#     width, height = image.size

#     # Calculate patch grid size (ViT usually has 14x14 or 7x7)
#     num_patches = attention_map.shape[1]
#     grid_size = int(np.sqrt(num_patches))

#     # Resize attention map to image size
#     attn_map_resized = attention_map.reshape(grid_size, grid_size)
#     attn_map_resized = np.kron(attn_map_resized, np.ones((height // grid_size, width // grid_size)))

#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.imshow(image)
#     ax.imshow(attn_map_resized, cmap='jet', alpha=0.5)
#     ax.set_title(title)
#     ax.axis('off')
#     plt.show()

# # Run t-SNE for visualization
# def visualize_tsne(features, labels):
#     tsne = TSNE(n_components=2, random_state=0)
#     tsne_results = tsne.fit_transform(features)

#     # Plot t-SNE
#     sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette="Set1")
#     plt.title("t-SNE visualization of feature space")
#     plt.show()