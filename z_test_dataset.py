import unittest
from dataprocessor import MyDatasetDoc
from dataprocessor.dataset import MyDatasetCorner, SmartDoc, SmartDocCorner

class DataDocTestCase(unittest.TestCase):
    def test_init_MyDataset(self):
        ds = MyDatasetDoc(['/home/mhadar/projects/doc_scanner/data/data_generator/v1'])
        ds = MyDatasetDoc('/home/mhadar/projects/doc_scanner/data/data_generator/v1')
        assert isinstance(ds.myData[0], list)
        assert ds.myData[1].shape[1] == 8
        assert len(ds.myData[0]) == len(ds.myData[1])
    
    def test_MyDataset_sandbox(self):
        ds = MyDatasetDoc('/home/mhadar/projects/doc_scanner/data/data_generator/sandbox')
        assert isinstance(ds.myData[0], list)
        assert ds.myData[1].shape[1] == 8
        assert len(ds.myData[0]) == 7
        assert len(ds.myData[1]) == 7
    
    def test_init_SmartDoc(self):
        ds = SmartDoc()

class DataCornerTestCase(unittest.TestCase):
    def test_init_MyDataset(self):
        ds = MyDatasetCorner(['/home/mhadar/projects/doc_scanner/data/data_generator/v1_corners'])
        ds = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/v1_corners')
        assert isinstance(ds.myData[0], list)
        assert ds.myData[1].shape[1] == 2
        assert len(ds.myData[0]) == len(ds.myData[1])
    
    def test_MyDataset_sandbox(self):
        ds = MyDatasetCorner('/home/mhadar/projects/doc_scanner/data/data_generator/sandbox_corners')
        assert isinstance(ds.myData[0], list)
        assert ds.myData[1].shape[1] == 2
        assert len(ds.myData[0]) == 5
        assert len(ds.myData[1]) == 5
    
    def test_init_SmartDoc(self):
        ds = SmartDocCorner()

if __name__ == '__main__':
    unittest.main()


