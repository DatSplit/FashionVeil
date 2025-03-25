# def transform(self, batch: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Preprocesses a batch of data for object detection training and validation.
#     HuggingFace datasets automatically batches data before transformation.

#     Args:
#         batch (Dict[str, Any]): A dictionary containing batched image and object data.

#     Returns:
#         Dict[str, Any]: A dictionary with transformed pixel values and labels.
#     """
#     # Process the entire batch at once - convert it to expected format
#     pixel_values_list = []
#     labels_list = []

#     for image_id in range(len(batch['image'])):

#         image = batch['image'][image_id]
#         image = fix_channels(ToTensor()(image))
#         pixel_values = self.feature_extractor(
#             [image], return_tensors='pt')['pixel_values']

#         bbox = [rescale_bboxes(batch['objects'][image_id]['bbox'],
#                                (batch['width'][image_id], batch['height'][image_id]))]
#         bbox = [xyxy_to_xcycwh(torch.Tensor(bbox_i)) for bbox_i in bbox]
#         class_labels = batch['objects'][image_id]['category']
#         # Create label for this image
#         label = {
#             "boxes": bbox[0] if bbox and len(bbox[0]) > 0 else torch.zeros((0, 4)),
#             "class_labels": torch.LongTensor(class_labels),
#             "image_id": torch.tensor([batch['image_id'][image_id]]).int(),
#             "area": torch.Tensor(batch['objects'][image_id]['area']),
#             "iscrowd": torch.zeros(len(class_labels)).int(),
#             "orig_size": torch.tensor([(batch['width'][image_id], batch['height'][image_id])]).int(),
#             "size": torch.tensor([pixel_values.shape[1:]]).int()[0],
#         }

#         pixel_values_list.append(pixel_values)
#         labels_list.append(label)

#     return {
#         "pixel_values": pixel_values_list,
#         "labels": labels_list
#     }

# def collate_fn(self, batch):
#     """
#     Collates batches for the dataloader.

#     Args:
#         batch: A list of dictionaries from the transform method

#     Returns:
#         Dict: A dictionary with batched pixel values and labels
#     """
#     pixel_values = []
#     labels = []

#     for item in batch:
#         if isinstance(item["pixel_values"], list):
#             for i, pv in enumerate(item["pixel_values"]):
#                 pixel_values.append(pv.squeeze(0))
#                 labels.append(item["labels"][i])
#         else:
#             pixel_values.append(item["pixel_values"].squeeze(0))
#             labels.append(item["labels"])

#     encoding = self.feature_extractor.pad(
#         pixel_values, return_tensors="pt")

#     collated_batch = {
#         'pixel_values': encoding['pixel_values'],
#         'labels': labels
#     }
#     return collated_batch
