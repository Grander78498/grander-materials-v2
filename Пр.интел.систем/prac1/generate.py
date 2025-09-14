groups = ['Queen', 'Scorpions', 'Led Zeppelin', 'Black Sabbath', 'Deep Purple',
          'Kendrick Lamar', 'Kanye West', 'Tyler, the Creator', 'Jay-Z', '2pac']
group_df = pd.DataFrame(index=groups, columns=groups)
popularities = []
for i in range(len(groups) // 5):
    popularities.extend(range(5, 0, -1))


for i in range(len(groups) // 5):
    for j in range(5 * i):
        for k in range(5 * i, 5 * i + 5):
            group_df.iloc[j, k] = group_df.iloc[k, j] = 1
    for j in range(5 * i, 5 * i + 5):
        for k in range(5 * i, 5 * i + 5):
            if j != k:
                group_df.iloc[j, k] = group_df.iloc[k, j] = 3
            else:
                group_df.iloc[j, k] = 0
    for j in range(5 * i + 5, len(groups)):
        for k in range(5 * i, 5 * i + 5):
            group_df.iloc[j, k] = group_df.iloc[k, j] = 3

for i in range(len(groups)):
    group_df.iloc[i] *= popularities[i]
group_df