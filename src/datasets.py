# import datasets
# from transformers import AutoFeatureExtractor, DetrImageProcessor 
# from utils import fix_channels, rescale_bboxes, xyxy_to_xcycwh
# import torch
# from torchvision.transforms import ToTensor, ToPILImage
# from torch.utils.data import DataLoader
# # 95% Percent of the dataset will be used for training
# train_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction("train",from_=0, to=95, unit="%", rounding="pct1_dropremainder"))
# # 5% of the dataset will be used for validation
# val_dataset = datasets.load_dataset("detection-datasets/fashionpedia", split=datasets.ReadInstruction("train",from_=95, to=100, unit="%", rounding="pct1_dropremainder"))
# cats = train_dataset.features['objects'].feature['category']

# def idx_to_text(indexes):
#     """
#     Converts an index into a category label.
#     :param indexes: List of indexes
#     :return: List of category labels
#     """
#     labels = []
#     for i in indexes:
#         labels.append(cats.names[i])
#     return labels
# feature_extractor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")

# def transform(batch):
#     inputs = {}
#     image = batch['image']
#     image = fix_channels(ToTensor()(image[0]))
#     inputs['pixel_values'] = feature_extractor([image], return_tensors='pt')['pixel_values']
#     labels = []
#     bbox = [rescale_bboxes(batch['objects'][i]['bbox'], (batch['width'][i], batch['height'][i])) for i in range(len(batch['objects']))]
#     bbox = [xyxy_to_xcycwh(torch.Tensor(bbox_i)) for bbox_i in bbox]
#     labels.append({
#         "boxes": bbox,
#         "class_labels": [object['category'] for object in batch['objects']],
#         "image_id": torch.Tensor([batch['image_id']]).int(),
#         "area": [object['area'] for object in batch['objects']],
#         "iscrowd": torch.Tensor([0 for _ in batch['objects']]).int(),
#         "orig_size": torch.Tensor([(batch['width'], batch['height'])]).int(),
#         "size": torch.Tensor([inputs['pixel_values'].shape[1:]])[0].int(),
#     })
#     inputs['labels'] = labels
#     return inputs
# prepared_train = train_dataset.with_transform(transform)
# prepared_val = val_dataset.with_transform(transform)
# def collate_fn(batch):
#     collated = {}
#     collated["pixel_values"] = feature_extractor.pad([item['pixel_values'] for item in batch], return_tensors="pt")['pixel_values']
#     collated["labels"] = []
#     for item in batch:
#         item['labels']['boxes'] = torch.stack(item['labels']['boxes'])[0]
#         item['labels']['area'] = torch.Tensor(item['labels']['area'])
#         item['labels']['class_labels'] = torch.Tensor(item['labels']['class_labels'])[0]
#         item['labels']['class_labels'] = item['labels']['class_labels'].type(torch.LongTensor)
#         collated["labels"].append(item['labels'])
#     return collated

# BATCH_SIZE = 1
# train_dataloader = DataLoader(prepared_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=27)
# val_dataloader = DataLoader(prepared_val, collate_fn=collate_fn, batch_size=BATCH_SIZE, num_workers=27)