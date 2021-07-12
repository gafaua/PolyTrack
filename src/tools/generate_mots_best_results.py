import pycocotools.mask as rletools
import pycocotools.coco as coco
import os
import numpy as np
from progress.bar import Bar

def generate_file():
    num_points = 48

    ann_path = f'../../data/MOTS/json_gt/val_{num_points}.json'

    anns = coco.COCO(ann_path)

    images_anns = {}

    i = 0
    im_id = -1
    for o in sorted(anns.anns.values(), key=lambda x: x['image_id']):
        if im_id != o['image_id']:
            im_id = o['image_id']
            i += 1
            images_anns[i] = []
        images_anns[i].append(o)


    with open('../../exp/tracking,polydet/MOTS20-09.txt', 'w') as f:
        bar = Bar(f'Saving best possible results for MOTS20-09', max=len(images_anns))

        for iid in images_anns:
            tracks_in_frame = []

            for item in images_anns[iid]:
                rle_mask = rletools.frPyObjects([item['poly']], 1080, 1920)[0]

                tracks_in_frame.append({
                'track_id': item['track_id'],
                'cat_id':  item['category_id'],
                'mask': rle_mask,
                'pseudo_depth': item['pseudo_depth']
                })

            if len(tracks_in_frame) > 1:
                # make sure no masks overlap in the same frame
                merged = np.ones((1080,1920), dtype=np.float) * -1
                tracks_in_frame.sort(key=lambda x: x['pseudo_depth'])

                for i, track in enumerate(tracks_in_frame):
                    binary_mask = rletools.decode(track['mask'])
                    merged *= np.logical_not(binary_mask)
                    merged += binary_mask * i
                
                for i, track in enumerate(tracks_in_frame):
                    binary_mask = merged == i
                    track['mask'] = rletools.encode(np.asfortranarray(binary_mask))

            for track in tracks_in_frame:
                # write line to file
                track_id = track['track_id']
                cat_id = track['cat_id']
                rle_mask = track['mask']['counts'].decode(encoding='UTF-8')

                f.write(f'{iid} {track_id} {cat_id} 1080 1920 {rle_mask}\n')
            Bar.suffix = f'[{iid}/{len(images_anns)}]|Tot: {bar.elapsed_td} |ETA: {bar.eta_td} |Tracks: {len(tracks_in_frame)}'
            bar.next()

generate_file()
os.system('python MOTS/evalMOTS.py --gt_dir ../../data/MOTS/ --res_dir ../../exp/tracking,polydet/ --seqmaps_dir ../../data/MOTS/seqmaps/')