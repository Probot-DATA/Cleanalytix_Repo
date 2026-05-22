import pandas as pd

from cleanalytix import Run_DQ_Pipeline


def main():
    df = pd.DataFrame(
        {
            "age": [25, None, 41, 41],
            "name": [" Alice", "Bob ", "Bob", None],
            "city": ["NY", "NY", "LA", "LA"],
        }
    )

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[df],
    )

    print(result["base_data"]["dirty_scores"])
    print(result["base_data"]["recommendations"].head())


if __name__ == "__main__":
    main()
