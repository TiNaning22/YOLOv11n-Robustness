from polygon_to_bbox import run

run(
    input_dir="labels/",
    output_dir="output/",
    consistent=True,
    method="median",
    class_names={0: "cat", 1: "dog"}
)