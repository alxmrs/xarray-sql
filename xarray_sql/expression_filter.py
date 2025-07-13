"""
Simple PyArrow expression filtering for xarray Datasets.

This module provides basic filtering capabilities by translating simple
PyArrow expressions to xarray operations.
"""

import re
import numpy as np
import xarray as xr
import pyarrow.compute as pc
from typing import Optional, Any


class SimpleExpressionFilter:
    """
    Simplified PyArrow expression filter for xarray Datasets.
    
    Supports basic comparison operations and field references.
    This is a minimal implementation for the adapter.
    """
    
    def __init__(self):
        """Initialize the filter."""
        pass
    
    def apply_filter(self, ds: xr.Dataset, expr: pc.Expression) -> xr.Dataset:
        """
        Apply a PyArrow expression to filter an xarray Dataset.
        
        Args:
            ds: xarray Dataset to filter
            expr: PyArrow expression for filtering
            
        Returns:
            Filtered xarray Dataset
        """
        # Convert expression to string and parse
        expr_str = str(expr)
        return self._parse_and_apply(ds, expr_str)
    
    def _parse_and_apply(self, ds: xr.Dataset, expr_str: str) -> xr.Dataset:
        """Parse expression string and apply to dataset."""
        # Remove PyArrow wrapper
        content = self._extract_content(expr_str)
        
        # Handle simple comparison expressions
        if self._is_simple_comparison(content):
            return self._apply_simple_comparison(ds, content)
        
        # Handle logical AND
        if ' and ' in content:
            return self._apply_logical_and(ds, content)
        
        # Handle logical OR  
        if ' or ' in content:
            return self._apply_logical_or(ds, content)
        
        # If we can't parse it, return original dataset with warning
        import warnings
        warnings.warn(f"Could not parse expression: {content}", UserWarning)
        return ds
    
    def _extract_content(self, expr_str: str) -> str:
        """Extract expression content from PyArrow string representation."""
        # Remove outer parentheses if present
        content = expr_str.strip()
        if content.startswith('(') and content.endswith(')'):
            content = content[1:-1]
        
        return content.strip()
    
    def _is_simple_comparison(self, content: str) -> bool:
        """Check if content is a simple comparison."""
        operators = ['==', '!=', '<=', '>=', '<', '>']
        return any(op in content for op in operators)
    
    def _apply_simple_comparison(self, ds: xr.Dataset, content: str) -> xr.Dataset:
        """Apply a simple comparison filter."""
        # Find operator
        operator = None
        for op in ['==', '!=', '<=', '>=', '<', '>']:
            if op in content:
                operator = op
                break
        
        if not operator:
            return ds
        
        # Split by operator
        parts = content.split(operator, 1)
        if len(parts) != 2:
            return ds
        
        left = parts[0].strip()
        right = parts[1].strip()
        
        # Extract field name and value
        field_name = self._extract_field_name(left)
        value = self._extract_value(right)
        
        if field_name is None or value is None:
            return ds
        
        # Apply filter to dataset
        return self._filter_dataset(ds, field_name, operator, value)
    
    def _apply_logical_and(self, ds: xr.Dataset, content: str) -> xr.Dataset:
        """Apply logical AND of multiple conditions."""
        # Remove outer parentheses if present
        content = content.strip()
        if content.startswith('(') and content.endswith(')'):
            content = content[1:-1]
        
        # Split by ' and ' at top level
        parts = self._split_by_operator(content, ' and ')
        
        # Apply each condition sequentially
        result = ds
        for part in parts:
            result = self._parse_and_apply(result, part.strip())
        
        return result
    
    def _apply_logical_or(self, ds: xr.Dataset, content: str) -> xr.Dataset:
        """Apply logical OR of multiple conditions."""
        # For OR operations, we need to apply each condition to original dataset
        # and then combine results - this is more complex for xarray
        # For now, just apply first condition with warning
        import warnings
        warnings.warn("OR operations not fully supported, applying first condition only", UserWarning)
        
        # Remove outer parentheses if present
        content = content.strip()
        if content.startswith('(') and content.endswith(')'):
            content = content[1:-1]
        
        # Split by ' or ' and take first part
        parts = self._split_by_operator(content, ' or ')
        if parts:
            return self._parse_and_apply(ds, parts[0].strip())
        
        return ds
    
    def _split_by_operator(self, content: str, operator: str) -> list[str]:
        """Split content by operator, respecting parentheses."""
        parts = []
        current = ""
        depth = 0
        i = 0
        
        while i < len(content):
            if content[i] == '(':
                depth += 1
                current += content[i]
            elif content[i] == ')':
                depth -= 1
                current += content[i]
            elif depth == 0 and content[i:i+len(operator)] == operator:
                parts.append(current)
                current = ""
                i += len(operator) - 1
            else:
                current += content[i]
            i += 1
        
        if current:
            parts.append(current)
        
        return parts
    
    def _extract_field_name(self, field_expr: str) -> Optional[str]:
        """Extract field name from field expression."""
        # Handle simple field names
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', field_expr):
            return field_expr
        
        # Handle quoted field names
        if field_expr.startswith('"') and field_expr.endswith('"'):
            return field_expr[1:-1]
        
        return None
    
    def _extract_value(self, value_expr: str) -> Optional[Any]:
        """Extract value from value expression."""
        value_expr = value_expr.strip()
        
        # Number (int or float)
        if re.match(r'^-?\d+$', value_expr):
            return int(value_expr)
        if re.match(r'^-?\d+\.\d*$', value_expr):
            return float(value_expr)
        
        # String literal
        if value_expr.startswith('"') and value_expr.endswith('"'):
            return value_expr[1:-1]
        
        # Boolean
        if value_expr.lower() == 'true':
            return True
        if value_expr.lower() == 'false':
            return False
        
        return None
    
    def _filter_dataset(self, ds: xr.Dataset, field_name: str, operator: str, value: Any) -> xr.Dataset:
        """Apply filter to dataset based on field, operator, and value."""
        # Check if field exists
        if field_name not in ds.coords and field_name not in ds.data_vars:
            return ds
        
        # Get the field data
        if field_name in ds.coords:
            field_data = ds.coords[field_name]
            # For coordinate filtering, select based on coordinate values
            if operator == '==':
                selection = {field_name: field_data[field_data == value]}
            elif operator == '!=':
                selection = {field_name: field_data[field_data != value]}
            elif operator == '<':
                selection = {field_name: field_data[field_data < value]}
            elif operator == '>':
                selection = {field_name: field_data[field_data > value]}
            elif operator == '<=':
                selection = {field_name: field_data[field_data <= value]}
            elif operator == '>=':
                selection = {field_name: field_data[field_data >= value]}
            else:
                return ds
                
            try:
                filtered_ds = ds.sel(selection)
                return filtered_ds
            except Exception:
                return ds
        else:
            # For data variable filtering, we need to find valid coordinates
            # This is more complex - we need to identify which coordinate combinations
            # satisfy the condition and then select only those
            field_data = ds[field_name]
            
            # Create condition based on operator
            if operator == '==':
                condition = field_data == value
            elif operator == '!=':
                condition = field_data != value
            elif operator == '<':
                condition = field_data < value
            elif operator == '>':
                condition = field_data > value
            elif operator == '<=':
                condition = field_data <= value
            elif operator == '>=':
                condition = field_data >= value
            else:
                return ds
            
            try:
                # Use where to identify valid locations, then create a mask
                # that we can use to filter the entire dataset
                
                # For now, let's use a simpler approach: 
                # Return the dataset with where applied, but acknowledge it has NaNs
                # The adapter will handle this by checking for valid data
                filtered_ds = ds.where(condition, drop=True)
                return filtered_ds
            except Exception:
                # If filtering fails, return original dataset
                return ds