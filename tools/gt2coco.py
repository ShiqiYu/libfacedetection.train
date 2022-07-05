import os
from PIL import Image 
import json
from tqdm import tqdm


def gt2coco(root, save_path):
    with open(os.path.join(root, 'wider_face_split', 'wider_face_val_bbx_gt.txt'), 'r') as f:
        infos = f.readlines()

    annos = []
    imgs = []
    anno_id = 0
    img_id = 0

    read_anno, read_num = False, False
    num_face_per_img = 0
    for it in tqdm(infos):
        it = it.rstrip()
        if read_anno:
            coords = [float(c) for c in it.split(' ')]
            x1, y1, w, h = coords[:4]
            # x2 = x1 + w
            # y2 = y1 + h
            area = w * h
            annos.append({
                "segmentation": [],
                "area": area,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [x1, y1, w, h],
                "category_id": 1,
                "id": anno_id,
                "ignore": 0})
            anno_id += 1
            num_face_per_img -= 1
            if num_face_per_img == 0:
                read_anno = False
                img_id += 1
        elif read_num:
            read_anno = True
            read_num = False
            num_face_per_img = int(it)
            if num_face_per_img != 0:
                imgs.append(img_info)
            else:
                print('woca')
        else:
            img_path = os.path.join(root, "WIDER_val", "images", it)
            read_num = True        
            width, height = Image.open(img_path).size
            img_info = {"file_name": it, "height": int(height), "width": int(width), "id": img_id} 


    categories = [{'name': k, 'id': v} for k, v in zip(['background', 'face'], range(2))]
    final_save = {"images": imgs, "annotations": annos, "categories": categories}
    with open(save_path, 'w') as f:
        json.dump(final_save, f)
    print('Save to {}'.format(save_path))
    print(f'imgs:\t{len(imgs)}\nannos:\t{len(annos)}')
    print(imgs[:3])
    print(annos[:3])

if __name__ == '__main__':
    root = './data/widerface'
    gt2coco(root=root, save_path='./data/widerface/trainset.json')
        
