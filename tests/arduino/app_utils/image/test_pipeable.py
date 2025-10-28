# SPDX-FileCopyrightText: Copyright (C) 2025 ARDUINO SA <http://www.arduino.cc>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from unittest.mock import MagicMock
from arduino.app_utils.image.pipeable import PipeableFunction


class TestPipeableFunction:
    """Test cases for the PipeableFunction class."""
    
    def test_init(self):
        """Test PipeableFunction initialization."""
        mock_func = MagicMock()
        pf = PipeableFunction(mock_func, 1, 2, kwarg1="value1")
        
        assert pf.func == mock_func
        assert pf.args == (1, 2)
        assert pf.kwargs == {"kwarg1": "value1"}
    
    def test_call_no_existing_args(self):
        """Test calling PipeableFunction with no existing args."""
        mock_func = MagicMock(return_value="result")
        pf = PipeableFunction(mock_func)
        
        result = pf(1, 2, kwarg1="value1")
        
        mock_func.assert_called_once_with(1, 2, kwarg1="value1")
        assert result == "result"
    
    def test_call_with_existing_args(self):
        """Test calling PipeableFunction with existing args."""
        mock_func = MagicMock(return_value="result")
        pf = PipeableFunction(mock_func, 1, kwarg1="value1")
        
        result = pf(2, 3, kwarg2="value2")
        
        mock_func.assert_called_once_with(1, 2, 3, kwarg1="value1", kwarg2="value2")
        assert result == "result"
    
    def test_call_kwargs_override(self):
        """Test that new kwargs override existing ones."""
        mock_func = MagicMock(return_value="result")
        pf = PipeableFunction(mock_func, kwarg1="old_value")
        
        result = pf(kwarg1="new_value", kwarg2="value2")
        
        mock_func.assert_called_once_with(kwarg1="new_value", kwarg2="value2")
        assert result == "result"
    
    def test_ror_pipe_operator(self):
        """Test right-hand side pipe operator (value | function)."""
        def add_one(x):
            return x + 1
        
        pf = PipeableFunction(add_one)
        result = 5 | pf
        
        assert result == 6
    
    def test_or_pipe_operator(self):
        """Test left-hand side pipe operator (function | function)."""
        def add_one(x):
            return x + 1
        
        def multiply_two(x):
            return x * 2
        
        pf1 = PipeableFunction(add_one)
        pf2 = PipeableFunction(multiply_two)
        
        # Chain: add_one | multiply_two
        composed = pf1 | pf2
        
        assert isinstance(composed, PipeableFunction)
        result = composed(5)  # (5 + 1) * 2 = 12
        assert result == 12
    
    def test_or_pipe_operator_with_non_callable(self):
        """Test pipe operator with non-callable returns NotImplemented."""
        pf = PipeableFunction(lambda x: x)
        with pytest.raises(TypeError, match="unsupported operand type"):
            pf | "not_callable"

    def test_repr_with_function_name(self):
        """Test string representation with function having __name__."""
        def test_func():
            pass
        
        pf = PipeableFunction(test_func)
        assert repr(pf) == "test_func()"
    
    def test_repr_with_args_and_kwargs(self):
        """Test string representation with args and kwargs."""
        def test_func():
            pass
        
        pf = PipeableFunction(test_func, 1, 2, kwarg1="value1", kwarg2=42)
        repr_str = repr(pf)
        
        assert "test_func(" in repr_str
        assert "1" in repr_str
        assert "2" in repr_str
        assert "kwarg1=value1" in repr_str
        assert "kwarg2=42" in repr_str
    
    def test_repr_with_partial_object(self):
        """Test string representation with functools.partial object."""
        from functools import partial
        
        def test_func(a, b):
            return a + b
        
        partial_func = partial(test_func, b=10)
        pf = PipeableFunction(partial_func)
        
        repr_str = repr(pf)
        assert "test_func" in repr_str or "partial" in repr_str
    
    def test_repr_with_callable_without_name(self):
        """Test string representation with callable without __name__."""
        class CallableClass:
            def __call__(self):
                pass
        
        callable_obj = CallableClass()
        pf = PipeableFunction(callable_obj)
        
        repr_str = repr(pf)
        assert "CallableClass" in repr_str


class TestPipeableFunctionIntegration:
    """Integration tests for the PipeableFunction class."""
    
    def test_real_world_data_processing(self):
        """Test pipeable with real-world data processing scenario."""
        def filter_positive(numbers):
            return [n for n in numbers if n > 0]
        def filtered_positive():
            return PipeableFunction(filter_positive)

        def square_all(numbers):
            return [n * n for n in numbers]
        def squared():
            return PipeableFunction(square_all)
        
        def sum_all(numbers):
            return sum(numbers)
        def summed():
            return PipeableFunction(sum_all)
        
        data = [-2, -1, 0, 1, 2, 3]
        
        # Pipeline: filter positive -> square    -> sum
        #           [1, 2, 3]       -> [1, 4, 9] -> 14
        result = data | filtered_positive() | squared() | summed()
        assert result == 14
    
    def test_error_handling_in_pipeline(self):
        """Test error handling within pipelines."""
        def divide_by(x, divisor):
            return x / divisor  # May raise ZeroDivisionError
        def divided_by(divisor):
            return PipeableFunction(divide_by, divisor=divisor)        

        def round_number(x, decimals=2):
            return round(x, decimals)
        def rounded(decimals=2):
            return PipeableFunction(round_number, decimals=decimals)
        
        # Test successful pipeline
        result = 10 | divided_by(3) | rounded(decimals=2)
        assert result == 3.33
        
        # Test error propagation
        with pytest.raises(ZeroDivisionError):
            10 | divided_by(0) | rounded()
