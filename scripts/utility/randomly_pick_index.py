import numpy as np


def randomly_select_fraction(total_rows, seed_num):
    """
    Randomly select a fraction of the dataset based on the dataset size.

    For datasets with:
    - Less than 200 rows: select 40% of the rows.
    - 200 to 999 rows: select 20% of the rows.
    - 1000 to 9999 rows: select 10% of the rows.
    - 10000 or more rows: select 2% of the rows.

    Args:
    total_rows (int): The number of rows in the dataset.

    Returns:
    list: A list of randomly selected indices.
    """

    # determine the fraction to select based on the dataset size
    if total_rows < 200:
        fraction = 0.40  # 40%
    elif 200 <= total_rows <= 999:
        fraction = 0.20  # 20%
    elif 1000 <= total_rows <= 9999:
        fraction = 0.10  # 10%
    else:
        fraction = 0.02  # 2%

    # calculate the number of rows to pick based on the fraction
    num_indices_to_pick = int(total_rows * fraction)

    # randomly select the indices without replacement
    np.random.seed(seed_num)  # To ensure reproducibility
    random_indices = np.random.choice(total_rows, size=num_indices_to_pick, replace=False)

    # Convert the array of indices to a list of integers
    random_indices_list = random_indices.tolist()

    return random_indices_list
