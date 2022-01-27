import os
import numpy as np
from PIL import Image

from evaluation import QudrilateralFinder

# imports from my other projects:
import os, sys
sys.path.append(os.path.abspath('z_ref_doc_scanner'))  #TODO - why importing fails if i put it under referneces folder?
from z_ref_doc_scanner.dataset import Dataset
from z_ref_doc_scanner.utils import draw_qudrilateral, IOU

v0 = [  'v0',
        'results/trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb',
        'results/trained_models/corner/26122021_corner_smartdoc/nonamecorner_resnet.pb'
        ]

v1 = [  'v1',
        'results/trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb',
        'results/trained_models/corner/17122022_corner_mycorners/nonamemy_corner_resnet.pb'
        ]

if __name__ == '__main__':
    v = v0
    data_type = 'high'
    output_path = f'results/{v[0]}/{data_type}'
    img_suffix = ''
    data_dir = f'z_ref_doc_scanner/data/self_collected/{data_type}-level-camera/stills/'
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset(data_dir, ignore=True)
    iou = []
    for i in range(len(ds)):
        im, quad_true = ds.readimage(i)
        img_name = ds.get_name(i)
        
        qf = QudrilateralFinder(v[1], v[2])
        quad_pred = qf.find_qudrilateral(im)
        
        oim = Image.fromarray(draw_qudrilateral(quad_pred, np.array(im)))
        oim.save(f'{output_path}/{img_name}{img_suffix}.jpg')
        iou.append(IOU(quad_true, quad_pred))
        
        print(f'{img_name} --- iou={iou[-1]:.02f}')
    #TODO - write log file with all the details.
    #TODO - add timing
    print(np.mean(np.array(iou)))

    # corners_extractor = evaluation.corner_extractor.CornerExtractor("../documentModelWell")
    # corner_refiner = evaluation.corner_refiner.CornerRefiner("../cornerModelWell")
    # test_set_dir = args.data_dir
    # iou_results = []
    # my_results = []
    # dataset_test = dataprocessor.dataset.SmartDocDirectories(test_set_dir)
    # for data_elem in dataset_test.myData:

    #     img_path = data_elem[0]
    #     # print(img_path)
    #     target = data_elem[1].reshape((4, 2))
    #     img_array = np.array(Image.open(img_path))
    #     computation_start_time = time.clock()
    #     extracted_corners = corners_extractor.get(img_array)
    #     temp_time = time.clock()
    #     corner_address = []
    #     # Refine the detected corners using corner refiner
    #     counter=0
    #     for corner in extracted_corners:
    #         counter+=1
    #         corner_img = corner[0]
    #         refined_corner = np.array(corner_refiner.get_location(corner_img, 0.85))

    #         # Converting from local co-ordinate to global co-ordinate of the image
    #         refined_corner[0] += corner[1]
    #         refined_corner[1] += corner[2]

    #         # Final results
    #         corner_address.append(refined_corner)
    #     computation_end_time = time.clock()
    #     print("TOTAL TIME : ", computation_end_time - computation_start_time)
    #     r2 = utils.intersection_with_corection_smart_doc_implementation(target, np.array(corner_address), img_array)
    #     r3 = utils.intersection_with_corection(target, np.array(corner_address), img_array)

    #     if r3 - r2 > 0.1:
    #         print ("Image Name", img_path)
    #         print ("Prediction", np.array(corner_address), target)
    #         0/0
    #     assert (r2 > 0 and r2 < 1)
    #     iou_results.append(r2)
    #     my_results.append(r3)
    #     print("MEAN CORRECTED JI: ", np.mean(np.array(iou_results)))
    #     print("MEAN CORRECTED MY: ", np.mean(np.array(my_results)))

    # print(np.mean(np.array(iou_results)))
