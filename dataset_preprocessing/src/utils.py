import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_json(file_path) -> dict:
    """
    Load a JSON file and return its content.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def get_objects_and_occlusion_levels(annotation_dict: dict) -> list[tuple[str, float]]:
    """
    Extract objects and their occlusion levels from an annotation dictionary.
    Args:
        annotation_dict (dict): A dictionary containing annotation data.
    Returns:
        list[tuple[str, float]]: A list of tuples where each tuple contains the object label
                                 and its occlusion level (0.0 - 1.0, denoting 0% - 100%).
    """

    objects: list = annotation_dict.get('shapes', [])
    objects_and_occlusion_levels = [
        (object['label'], float(object.get('description') or '0.0')) for object in objects]

    return objects_and_occlusion_levels


def generate_histogram(data: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, name_to_save: str, rotation: int, show_labels: bool = True) -> None:
    colors = {
        "primary_dark": "#323a79",
        "primary_light": "#5EAADA",
        "white": "#ffffff",
    }
    sns.set_style("white")

    plt.figure(figsize=(20, 15))
    bar_plot = sns.barplot(
        data=data,
        x=x,
        y=y,
        color=colors["primary_dark"],

    )

    # Add value labels on top of the bars
    if show_labels:
        for p in bar_plot.patches:
            bar_plot.annotate(f'{int(p.get_height())}',
                              (p.get_x() + p.get_width() / 2., p.get_height()),
                              ha='center', va='bottom',
                              color=colors["primary_light"],
                              fontsize=16, fontweight='bold')

    # Title and axis formatting
    plt.title(title, fontsize=30, color=colors["primary_dark"])
    plt.xlabel(xlabel, fontsize=20, color=colors["primary_dark"])
    plt.ylabel(ylabel, fontsize=20, color=colors["primary_dark"])
    bar_plot.tick_params(colors=colors["primary_dark"])
    plt.xticks(fontsize=18, rotation=rotation, ha='center')
    plt.yticks(fontsize=18)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig(f'{name_to_save}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
