import pandas as pd

from cleanalytix import Run_DQ_Pipeline


def valid_age(value):
    if pd.isna(value):
        return True
    try:
        return 0 <= float(value) <= 120
    except (TypeError, ValueError):
        return False


def main():
    base = pd.DataFrame(
        {
            "age": [25, None, 130, 40],
            "name": [" Alice", "Bob ", "Bob", None],
            "city": ["NY", "NY", "LA", "SF"],
            "income": [50000, 52000, 999999, 48000],
        }
    )
    prod = pd.DataFrame(
        {
            "age": [26, 39, 141, None],
            "name": ["Alice ", " Bob", "Bob", "Tom"],
            "city": ["NY", "LA", "LA", "SF"],
            "income": [51000, 49500, 1000000, 47000],
        }
    )

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
        new_dataset_list=[prod],
        rules={"age": valid_age},
        cleaning=True,
        interactive=False,
    )

    print(result["base_data"]["dirty_scores"])
    print(result["base_data"]["cleaned_scores"])
    print(result["prod_data"]["dirty_scores"])
    print(result["prod_data"]["change_log"].head())


if __name__ == "__main__":
    main()
