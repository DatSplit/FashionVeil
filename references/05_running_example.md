#### Example: reproducing all results for `RFDETR-B`

First download `Fashionpedia` formatted for rfdetr:

```bash
curl -L "https://fashionpedia-rfdetr.s3.eu-central-1.amazonaws.com/rfdetr_fashionpedia.zip?response-content-disposition=inline&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEN7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDGV1LWNlbnRyYWwtMSJHMEUCIQDETe3zPsVkfezqCOG472YxWuAX30gJ%2FwWaVeH5eO%2FlYgIgA%2FyJfUde7wU%2FsmYaZ0N0F%2FtY6vjRWgViXioPd7YsfMwqvgMI2P%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw5NDEzNzcxMTU4MDYiDJxKU01m6yTRgqjuXiqSAy3CKDhSepHaMeLFYKzD390oLjzHk3Bmbe4xXkwVdPgUBnr6xssi9mU13NghdNpVfxsJSN%2BkkQkkZ8uHpTJSBdBzGO%2F0V3BnhLSpVC6qmSPTRgNlXd06xLXDyZIgJugU5uDMzsMSgZ%2FQiHXSMYLe9iSPKqXKK2pTpDE%2FAl9lv20mOauAZ9xiPdtXYpMrCW8ZGqTFESC39T1aNF8Zv7r0Mkx%2FJfEVQjpE29HAfDXS2QyG2aHC4a92xt%2BmR9JBIwZtScHhC2X8E3%2Bfq33PM3TpcrKytII0BvkJ%2BVvhdYQqPZw68%2FBxm37JZZYUuaD42pS%2Fiz9ZyuyafRwozhalr7RPZBZarq6z33cgQwz4ePnlhNb7%2FnEclHx0BJ0NoWk2ahWNahE25%2FQrpawYEQ%2BqsHqXgey1u3RVkF7C6J4duBdPxj88kaykzpTU4zAzU8z8v9cvCsnceaWgtH1mEm2n5O8mSM2luZW2HR85sS8Oc2SRmg5gTbYygAGFbrNtz0oNJLmk4Dgpbu%2Bh9lPjoruAruRShro%2FQTDL3o%2FDBjreAguM8wDv8Wx%2FLgWhgd70OQu03TU31y4%2Fs0bjhbP9ts2beQw0J9azvkxbJ1SxuQRAHPef30iEXpF3BeK2y6apmnDq5i6TlbJQwkLaNTOA9iG3qw%2BWzPUtsGCDs2p3TPNo5n3GppmSkeJJh2nKE0W3%2Buj1W3A6GDkC%2FYrO5lmm9Kz43yIXNkQxB1e1q%2B6SdgqnHMIWoyKbOMxYwp%2BPkg1GFaXyPjMmu1BS3Wim8LU8le6Zqm26OEc1tmjZwBu2L6BV%2FkC50Fu6R2nZ%2BEVJXBZA3%2BUpqlCfCcdHFvCsZYQ2ptS66GhHwqDkscVQafzFfd8IeY71zUmmrvw0Ty6xM27YyL0xTnwLopitNBOZy4zAvR0lqPKxLDtDV3101OlWicPCBJoA3tsFMLvvTyS1yqAdf4TRbAXZ9mfHWacjoklMk3G66Gk3wh%2B6swK7PJRmimNqSvkGhYSq5%2BdaNfukuTSZ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIA5WLTS22PPFIJBTL5%2F20250701%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20250701T142403Z&X-Amz-Expires=300&X-Amz-SignedHeaders=host&X-Amz-Signature=16771766cb335bfac727960cc01029a7fe0345a949f744c8468eaa5f1d03d495" -o fashionpedia_rfdetr.zip
unzip -o fashionpedia_rfdetr.zip
cd models/rfdetr
python3 train_rfdetr.py \
    --onnx_path "" \
    --dataset_dir "~/FashionVeil/rfdetr_fashionpedia/" \
    --model_type "base" \
    --output_dir "rfdetr_base_results" \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 8 \
    --lr 1e-4 \
    --resolution 1120 

```