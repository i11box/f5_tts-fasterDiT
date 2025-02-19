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
        result = cm1.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['0.000']

    def test_get_need_cal_window_res_2(self, compress_manager):
        # 测试用例2：none后面跟none，第二个none后面有wars
        cm2 = compress_manager
        cm2.compress_dict = {
            '0.000': 'none',
            '1.000': 'none',
            '2.000': 'wars'
        }
        result = cm2.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['1.000']

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
        result = cm3.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['0.000', '3.000']

    def test_get_need_cal_window_res_4(self, compress_manager):
        # 测试用例4：none后面没有wars
        cm4 = compress_manager
        cm4.compress_dict = {
            '0.000': 'none',
            '1.000': 'ast',
            '2.000': 'none'
        }
        result = cm4.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == []

    def test_get_need_cal_window_res_5(self, compress_manager):
        # 测试用例5：没有none的情况
        cm5 = compress_manager
        cm5.compress_dict = {
            '0.000': 'wars',
            '1.000': 'ast',
            '2.000': 'wars'
        }
        result = cm5.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == []

    def test_get_need_cal_window_res_6(self, compress_manager):
        # 测试用例6：边界情况 - 只有一个时间步
        cm6 = compress_manager
        cm6.compress_dict = {
            '0.000': 'none'
        }
        result = cm6.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == []

    def test_get_need_cal_window_res_7(self, compress_manager):
        # 测试用例7：none后面同时有wars和none
        cm7 = compress_manager
        cm7.compress_dict = {
            '0.000': 'none',
            '1.000': 'wars',
            '2.000': 'none',
            '3.000': 'wars'
        }
        result = cm7.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['0.000', '2.000']
        
    def test_get_need_cal_window_res_8(self, compress_manager):
        cm8 = compress_manager
        cm8.compress_dict = {
        "0.000": "none",
        "0.001": "ast",
        "0.005": "ast",
        "0.012": "wars",
        "0.021": "none",
        "0.032": "none",
        "0.046": "none",
        "0.062": "none",
        "0.081": "none",
        "0.102": "none",
        "0.25": "none",
        "0.151": "none",
        "0.179": "asc",
        "0.209": "asc",
        "0.241": "asc-wars",
        "0.275": "asc-wars",
        "0.311": "asc-wars",
        "0.349": "asc-wars",
        "0.388": "asc-wars",
        "0.428": "asc-wars",
        "0.471": "asc-wars",
        "0.515": "asc-wars",
        "0.560": "asc-wars",
        "0.606": "asc-wars",
        "0.653": "asc-wars",
        "0.700": "asc-wars",
        "0.749": "none",
        "0.799": "none",
        "0.849": "none",
        "0.899": "none",
        "0.950": "none",
        "1.000": "asc-wars"
        }
        
        result = cm8.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['0.000', '0.209', '0.950']
        
    def test_get_need_cal_window_res_9(self, compress_manager):
        cm9 = compress_manager
        cm9.compress_dict = {
            "0.000": "none",
            "0.001": "ast",
            "0.005": "none",
            "0.012": "none",
            "0.021": "none",
            "0.032": "none",
            "0.046": "none",
            "0.062": "none",
            "0.081": "none",
            "0.102": "none",
            "0.25": "none",
            "0.151": "none",
            "0.179": "none",
            "0.209": "none",
            "0.241": "none",
            "0.275": "asc-wars",
            "0.311": "asc-wars",
            "0.349": "asc-wars",
            "0.388": "asc-wars",
            "0.428": "asc-wars",
            "0.471": "asc-wars",
            "0.515": "asc-wars",
            "0.560": "none",
            "0.606": "none",
            "0.653": "none",
            "0.700": "none",
            "0.749": "none",
            "0.799": "none",
            "0.849": "none",
            "0.899": "none",
            "0.950": "none",
            "1.000": "none"
        }
    
        result = cm9.get_need_cal_window_res()
        assert [k for k, v in result.items() if v] == ['0.241']