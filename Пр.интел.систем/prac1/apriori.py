results: list[list[set]] = []
while True:
    f_k: list[set] = []
    if k == 1:
        for item in unique_items:
            if norm_df.sum()[item] >= f:
                f_k.append({item})
    else:
        if k == 2:
            combs: list[set] = [x[0].union(x[1]) for x in combinations(results[0], r=2)]
        else:
            previous = results[k - 2]
            combs = []
            for i, elem in tqdm(enumerate(previous)):
                _set = set(sorted(elem)[:-1])
                if any(_set.issubset(x) for x in combs):
                    continue
                for j, elem1 in enumerate(previous):
                    if i == j:
                        continue
                    if _set.issubset(elem1) and all(any(len(set(subset).difference(x)) == 0 for x in previous) for subset in combinations(elem.union(elem1), r=k-1)):
                        combs.append(elem.union(elem1))
        
        for elem in combs:
            cnt = count_rows(list(elem))
            if cnt >= f:
                f_k.append(elem)
    
    if len(f_k) == 0:
        break
    results.append(f_k)
    k += 1