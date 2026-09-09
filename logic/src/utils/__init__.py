"""
Utilities package for WSmart+ Route logic.

Contains helper modules for:
- Argument parsing and configuration loading.
- Input/Output and logging.
- Mathematical operations and graph utilities.
- Data processing and visualization.
- Environment and model setup.

Attributes:
    actions: Tensor actions/operations utilities
    configs: Configuration utilities
    data: Data processing utilities
    decoding: Decoding utilities
    docs: Documentation utilities
    functions: Function utilities
    graph: Graph utilities
    io: Input/Output utilities
    model: Model utilities
    output: Output utilities
    plotting: Visualization and plotting utilities
    policy: Policy utilities
    security: Security utilities
    tasks: Task utilities
    ui: Streamlit User Interface (UI) utilities
    validation: Validation utilities

Example:
    >>> from logic.src.utils import read_json
    >>> data = read_json("data.json")
    >>> zip_directory("data", "data.zip")
    >>> extract_zip("data.zip", "data")
    >>> confirm_proceed("Are you sure?")
    True
    >>> compose_dirpath("data", "data.json")
    "data/data.json"
    >>> split_file("data.json", 10)
    >>> chunk_zip_content("data.zip", 10)
    >>> reassemble_files(["data1.json", "data2.json"])
    >>> process_dict_of_dicts({"a": {"b": 1}})
    {"a": {"b": 1}}
    >>> process_list_of_dicts([{"a": 1}, {"a": 2}])
    [{"a": 1}, {"a": 2}]
    >>> process_dict_two_inputs({"a": {"b": 1}, "c": {"d": 2}})
    {"a": {"b": 1}, "c": {"d": 2}}
    >>> process_list_two_inputs([{"a": 1}, {"a": 2}], [{"b": 3}, {"b": 4}])
    [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    >>> find_single_input_values([{"a": 1}, {"a": 2}])
    [1, 2]
    >>> find_two_input_values([{"a": 1}, {"a": 2}], [{"b": 3}, {"b": 4}])
    [(1, 3), (2, 4)]
    >>> process_pattern_files("data/*.json")
    >>> process_file("data.json")
    >>> process_pattern_files_statistics("data/*.json")
    >>> process_file_statistics("data.json")
    >>> preview_changes([{"a": 1}, {"a": 2}])
    >>> preview_file_changes("data.json")
    >>> preview_pattern_files_statistics("data/*.json")
    >>> preview_file_statistics("data.json")
    >>> read_output("output.json")
    {"a": 1}
"""
