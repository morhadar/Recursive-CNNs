import unittest
from PIL import Image
from torchvision import transforms
from dataprocessor import MyDatasetDoc
from dataprocessor.dataset import MyDatasetCorner, SmartDoc, SmartDocCorner

from utils import draw_circle_pil

class DataDoc_TestCase(unittest.TestCase):
    def test_init_MyDataset(self):
        ds = MyDatasetDoc('/home/mhadar/projects/doc_scanner/data/data_generator/v1')
        assert isinstance(ds.data, list)
        assert ds.target.shape[1] == 8
        assert len(ds.data) == len(ds.target)
    
    def test_MyDataset_sandbox(self):
        ds = MyDatasetDoc('/home/mhadar/projects/doc_scanner/data/data_generator/sandbox')
        im, target = ds[0]
        assert isinstance(ds.data, list)
        assert ds.target.shape[1] == 8
        assert len(ds.data) == 7
        assert len(ds.target) == 7
    
    def test_init_SmartDoc(self):
        ds = SmartDoc()

class DataCorner_TestCase(unittest.TestCase):
    @staticmethod
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def test_init_MyDataset(self):
        ds = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/v1_corners')
        assert isinstance(ds.data, list)
        assert ds.target.shape[1] == 2
        assert len(ds.data) == len(ds.target)
    
    def test_MyDataset_sandbox(self):
        ds = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/sandbox_corners')
        im, target = ds[0]
        assert isinstance(ds.data, list)
        assert ds.target.shape[1] == 2
        assert len(ds.data) == 5
        assert len(ds.target) == 5
    
    def test_rotation(self):
        ds = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/v2_corners')
        ds_rotated = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/v2_corners', is_rotating=True)
        trans_to_pil = transforms.ToPILImage()
        for i in range(4):
            orig, target_orig = ds[i]   
            rotated, target_rotated = ds_rotated[i]
            orig, rotated = trans_to_pil(orig), trans_to_pil(rotated)
            draw_circle_pil(orig, (target_orig * orig.size).astype(int), radious=5, outline='yellow', width=2)
            draw_circle_pil(rotated, (target_rotated * rotated.size).astype(int), radious=5, outline='yellow', width=2)
            t = self.get_concat_h(orig, rotated)

            t.save(f'results/debug/{i}_rotated.jpg')       

    def test_init_SmartDoc(self):
        ds = SmartDocCorner()

if __name__ == '__main__':
    unittest.main()


