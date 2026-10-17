"""Complete deterministic query and bind example; run with the Kedi interpreter."""

import kedi


@kedi.query
def item_count(items: list[str]) -> int:
    """kedi
    = `len(items)`
    """
    ...


@kedi.bind(file="bound.kedi")
def total(values: list[int]) -> int: ...


if __name__ == "__main__":
    assert item_count(["parser", "runtime"]) == 2
    assert total([2, 3, 5]) == 10
    print("query=2, bind=10")
