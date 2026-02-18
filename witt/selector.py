"""
Selector system for addressing ranges of tokens and layers.

Selectors wrap whatever the user passes to __getitem__ (int, slice, list)
and resolve it to a torch-compatible index at evaluation time.
"""
from typing import Union, List


class Selector:
    """
    Base class for dimension selectors.
    
    A Selector stores the user's indexing key and resolves it to
    concrete indices when the actual dimension size is known.
    """

    @staticmethod
    def from_key(key: Union[int, slice, list, tuple, 'Selector']) -> 'Selector':
        """Create a Selector from a __getitem__ key."""
        if isinstance(key, Selector):
            return key
        if isinstance(key, int):
            return IndexSelector(key)
        if isinstance(key, slice):
            return SliceSelector(key)
        if isinstance(key, (list, tuple)):
            return ListSelector(list(key))
        raise TypeError(f"Unsupported selector key type: {type(key)}")

    def resolve(self, dim_size: int):
        """
        Return a value usable as tensor[:, <here>, :].
        
        For IndexSelector this is an int (squeezes the dim).
        For SliceSelector this is a slice.
        For ListSelector this is a list of ints.
        """
        raise NotImplementedError

    @property
    def is_single(self) -> bool:
        """True if this selects exactly one element (squeezes the dimension)."""
        raise NotImplementedError

    @property
    def count(self) -> int:
        """
        Return the number of elements selected.
        Only valid for non-single selectors (SliceSelector, ListSelector).
        For IndexSelector, this is 1.
        """
        raise NotImplementedError

    def indices(self, dim_size: int) -> List[int]:
        """Return the concrete list of selected indices."""
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class IndexSelector(Selector):
    """Selects a single element by index. Squeezes the dimension."""

    def __init__(self, index: int):
        self._index = index

    def resolve(self, dim_size: int) -> int:
        idx = self._index
        if idx < 0:
            idx += dim_size
        if idx < 0 or idx >= dim_size:
            raise IndexError(
                f"Index {self._index} out of range for dimension of size {dim_size}"
            )
        return idx

    @property
    def is_single(self) -> bool:
        return True

    @property
    def count(self) -> int:
        return 1

    def indices(self, dim_size: int) -> List[int]:
        return [self.resolve(dim_size)]

    def __repr__(self) -> str:
        return f"IndexSelector({self._index})"


class SliceSelector(Selector):
    """Selects a range of elements via a slice. Preserves the dimension."""

    def __init__(self, slc: slice):
        self._slice = slc

    def resolve(self, dim_size: int) -> slice:
        # Return a normalized slice with concrete start/stop/step
        start, stop, step = self._slice.indices(dim_size)
        return slice(start, stop, step)

    @property
    def is_single(self) -> bool:
        return False

    @property
    def count(self) -> int:
        """
        Compute the number of elements this slice selects.
        
        For slices with known start/stop (not None), we can compute this.
        For open slices (e.g. slice(None) = ':'), we raise since
        the count depends on dim_size. Use indices() instead.
        """
        start = self._slice.start or 0
        stop = self._slice.stop
        step = self._slice.step or 1
        if stop is None:
            raise ValueError(
                "Cannot determine count for open-ended slice without dim_size. "
                "Use indices(dim_size) or count_for(dim_size) instead."
            )
        # Calculate length like range(start, stop, step)
        if step > 0:
            return max(0, (stop - start + step - 1) // step)
        else:
            return max(0, (start - stop - step - 1) // (-step))

    def count_for(self, dim_size: int) -> int:
        """Compute count given a known dimension size."""
        return len(range(*self._slice.indices(dim_size)))

    def indices(self, dim_size: int) -> List[int]:
        return list(range(*self._slice.indices(dim_size)))

    def __repr__(self) -> str:
        parts = []
        if self._slice.start is not None:
            parts.append(str(self._slice.start))
        parts.append(':')
        if self._slice.stop is not None:
            parts.append(str(self._slice.stop))
        if self._slice.step is not None:
            parts.append(f':{self._slice.step}')
        return f"SliceSelector({''.join(parts)})"


class ListSelector(Selector):
    """Selects arbitrary elements by a list of indices. Preserves the dimension."""

    def __init__(self, indices_list: List[int]):
        if not indices_list:
            raise ValueError("ListSelector requires at least one index")
        self._indices = indices_list

    def resolve(self, dim_size: int) -> List[int]:
        resolved = []
        for idx in self._indices:
            if idx < 0:
                idx += dim_size
            if idx < 0 or idx >= dim_size:
                raise IndexError(
                    f"Index {idx} out of range for dimension of size {dim_size}"
                )
            resolved.append(idx)
        return resolved

    @property
    def is_single(self) -> bool:
        return False

    @property
    def count(self) -> int:
        return len(self._indices)

    def indices(self, dim_size: int) -> List[int]:
        return self.resolve(dim_size)

    def __repr__(self) -> str:
        return f"ListSelector({self._indices})"
