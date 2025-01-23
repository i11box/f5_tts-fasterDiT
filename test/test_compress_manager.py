import pytest
from f5_tts.model.utils import CompressManager


class TestCompressManager:
    @pytest.fixture
    def compress_manager(self):
        return CompressManager()

    def test_get_need_cal_window_res_1(self, compress_manager):
        # 测试用例1：基本情况 - none后面直接跟wars
        cm1 = compress_manager
        cm1.compress_dict = {
            '0.000': 'none',
            '1.000': 'wars',
            '2.000': 'ast'
        }
        assert cm1.get_need_cal_window_res() == ['0.000']

    def test_get_need_cal_window_res_2(self, compress_manager):
        # 测试用例2：none后面跟none，第二个none后面有wars
        cm2 = compress_manager
        cm2.compress_dict = {
            '0.000': 'none',
            '1.000': 'none',
            '2.000': 'wars'
        }
        assert cm2.get_need_cal_window_res() == ['1.000']

    def test_get_need_cal_window_res_3(self, compress_manager):
        # 测试用例3：复杂情况 - 多个none和wars交替
        cm3 = compress_manager
        cm3.compress_dict = {
            '0.000': 'none',
            '1.000': 'wars',
            '2.000': 'ast',
            '3.000': 'none',
            '4.000': 'wars'
        }
        assert cm3.get_need_cal_window_res() == ['0.000', '3.000']

    def test_get_need_cal_window_res_4(self, compress_manager):
        # 测试用例4：none后面没有wars
        cm4 = compress_manager
        cm4.compress_dict = {
            '0.000': 'none',
            '1.000': 'ast',
            '2.000': 'none'
        }
        assert cm4.get_need_cal_window_res() == []

    def test_get_need_cal_window_res_5(self, compress_manager):
        # 测试用例5：没有none的情况
        cm5 = compress_manager
        cm5.compress_dict = {
            '0.000': 'wars',
            '1.000': 'ast',
            '2.000': 'wars'
        }
        assert cm5.get_need_cal_window_res() == []

    def test_get_need_cal_window_res_6(self, compress_manager):
        # 测试用例6：边界情况 - 只有一个时间步
        cm6 = compress_manager
        cm6.compress_dict = {
            '0.000': 'none'
        }
        assert cm6.get_need_cal_window_res() == []

    def test_get_need_cal_window_res_7(self, compress_manager):
        # 测试用例7：none后面同时有wars和none
        cm7 = compress_manager
        cm7.compress_dict = {
            '0.000': 'none',
            '1.000': 'wars',
            '2.000': 'none',
            '3.000': 'wars'
        }
        assert cm7.get_need_cal_window_res() == ['0.000', '2.000']
    