min_support = 4
start_point = 0
k = 2
while True:
    end_point = len(eclat_df)
    
    new_dict = {'Item': [], 'Transactions': []}
    for i in tqdm(range(start_point, end_point)):
        for j in range(i + 1, end_point):
            item1, item2 = eclat_df.iloc[i]['Item'], eclat_df.iloc[j]['Item']
            if k > 2 and not all(x == y for x in item1[:k - 2] for y in item2[:k - 2]):
                break

            tr1, tr2 = eclat_df.iloc[i]['Transactions'], eclat_df.iloc[j]['Transactions']
            if len(set(item1).union(set(item2))) == k:
                new_dict['Item'].append(sorted(set(item1).union(set(item2))))
                new_dict['Transactions'].append(list(set(tr1).intersection(set(tr2))))
    
    eclat_df = pd.concat([
        eclat_df,
        pd.DataFrame(new_dict)
    ], ignore_index=True)
    if len(eclat_df) == end_point:
        break
    k += 1
    start_point = end_point
    eclat_df = eclat_df[eclat_df['Transactions'].apply(lambda t: len(t) >= min_support)]
