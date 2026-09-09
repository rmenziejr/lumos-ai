# Bias Analysis

Use `bias_report()` to compare model behavior across protected-attribute groups.

## Limit high-cardinality numeric groups

Numeric protected attributes can create an unusable number of groups when every distinct value is treated separately. `bias_report()` now accepts `max_bins`, which defaults to `10`.

```python
from lumosai.model import bias_report

result = bias_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    protected_attribute=["age"],
    max_bins=5,
)
```

When a numeric protected attribute has more than `max_bins` distinct non-null values, Lumos groups it into at most that many equal-width bins. Numeric attributes with `max_bins` or fewer distinct values remain discrete groups.

Set `max_bins=None` to preserve every numeric value as its own group:

```python
result = bias_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    protected_attribute=["age"],
    max_bins=None,
)
```

Explicit bin edges always take precedence over automatic binning:

```python
result = bias_report(
    scored_frame,
    target="actual",
    prediction="prediction",
    protected_attribute={"age": [0, 18, 35, 50, 65, 120]},
    max_bins=3,
)
```

The explicit age bands above are used even though `max_bins=3`.

## Metric-gap visualization

Bias report HTML renders metric gaps as horizontal bars. The chart height grows with the number of metric/group comparisons, which keeps group labels readable instead of forcing a very wide plot with rotated labels.
