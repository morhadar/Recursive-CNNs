import unittest
from dataprocessor import MyDatasetDoc
from dataprocessor.dataset import SmartDoc

class DataMyDatasetDocTestCase(unittest.TestCase):
    def test_init_MyDataset(self):
        # ds = MyDatasetDoc()
        ds = MyDatasetDoc(['/home/mhadar/projects/doc_scanner/data/data_generator/v1'])
        ds = MyDatasetDoc('/home/mhadar/projects/doc_scanner/data/data_generator/v1')
        assert isinstance(ds.myData[0], list)
        assert ds.myData[1].shape[1] == 8
        assert len(ds.myData[0]) == len(ds.myData[1])
    
    def test_init_SmartDoc(self):
        ds = SmartDoc()


if __name__ == '__main__':
    unittest.main()


