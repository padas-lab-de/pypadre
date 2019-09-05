def _sub_list(l, start=-1, count=9999999999999):
    start = max(start, 0)
    stop = min(start + count, len(l))
    if start >= len(l):
        return []
    else:
        return l[start:stop]