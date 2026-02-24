def build_full_year_week_map(full_hours, n_weeks=52):
    """
    Assigns each hour in the full 8760-hour year to a fixed TargetX week label.
    Always splits into exactly 52 periods.

    Parameters:
    -----------
    full_hours : list
        The full list of hour labels, e.g. ["T001", ..., "T8760"]

    Returns:
    --------
    dict : {hour_label: week_label}
    """
    assert len(full_hours) == 8760, "Expected full 8760-hour time index"

    base = len(full_hours) // n_weeks  # 168
    remainder = len(full_hours) % n_weeks  # 8760 % 52 = 24

    week_map = {}
    idx = 0
    for w in range(n_weeks):
        length = base + (1 if w < remainder else 0)
        label = f"Target{w+1}"
        for _ in range(length):
            week_map[full_hours[idx]] = label
            idx += 1
    return week_map