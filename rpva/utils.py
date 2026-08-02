def require_columns(rows, columns):
    missing = [c for c in columns if any(c not in row for row in rows)]
    if missing:
        raise ValueError(f'Missing required columns: {sorted(set(missing))}')
