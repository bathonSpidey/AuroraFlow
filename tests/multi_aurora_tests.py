import unittest
from src.MultiRag.MultiAuroraRag import MultiAuroraRag
import os
class MyTestCase(unittest.TestCase):
    def test_multi_aurora_rag(self):
        rag = MultiAuroraRag(os.getcwd() + "/data/rice", "all-mpnet-base-v2")
        result = rag.search("pests of rice")
        rag.display_results(result)
        self.assertEqual(len(result), 5)


if __name__ == '__main__':
    unittest.main()
