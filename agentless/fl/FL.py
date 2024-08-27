from abc import ABC, abstractmethod

from openai import OpenAI

from agentless.repair.repair import construct_topn_file_context
from agentless.util.compress_file import get_skeleton
from agentless.util.postprocess_data import extract_code_blocks, extract_locs_for_files
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

MAX_CONTEXT_LENGTH = 128000


class FL(ABC):
    def __init__(self, instance_id, structure, problem_statement, **kwargs):
        self.structure = structure
        self.instance_id = instance_id
        self.problem_statement = problem_statement

    @abstractmethod
    def localize(self, top_n=1, mock=False) -> tuple[list, list, list, any]:
        pass


class LLMFL(FL):
    obtain_relevant_files_prompt = """
    Please look through the following GitHub problem description and Repository structure and provide a list of files that one would need to edit to fix the problem.

    ### GitHub Problem Description ###
    {problem_statement}

    ###

    ### Repository Structure ###
    {structure}

    ###

    Please return only the full file paths that exists in "Repository Structure", with no additional text or explanation. Return at most 10 files, but at least 3 files.
    Exclude any files that is not in "Repository Structure". The returned files should be separated by new lines ordered by most to least important and wrapped with ```
    For example:
    ```
    file1.py
    file2.py
    ```
    """

    refine_files_prompt = """
    ### Initially Identified Files (with functions/classes/variables) ###
    {initial_files}
    You previously selected the above "Initially Identified Files" as most relevant to fixing the issue.
    Now, considering the functions, classes and variables that might need modification in each files based on the GitHub problem description and Repository Structure, please refine the list.

    ### GitHub Problem Description ###
    {problem_statement}

    ### Repository Structure ###
    {structure}

    Remove the files that not in "Repository Structure". Please return at least 3 full paths that exist in "Repository Structure", separated by new lines ordered by most to least important, wrapped with ```.
    For example:
    ```
    file1.py
    file3.py
    ```
    """

    def get_detailed_file_info(self, filepath, structure):
        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            structure
        )

        # Find the file in the structure and retrieve detailed info
        detailed_info = []
        for file_content in files:
            if file_content[0] == filepath:
                detailed_info.append(f"File: {filepath}")
                for cls in classes:
                    if cls["file"] == filepath:
                        detailed_info.append(f"class: {cls['name']}")
                for func in functions:
                    if func["file"] == filepath:
                        detailed_info.append(f"function: {func['name']}")
                break

        return detailed_info

    def refine_file_selection(self, found_files, structure, problem_statement):
        # Gather function, class, and variable details for the initially identified files
        detailed_files = []
        for filepath in found_files:
            detailed_info = self.get_detailed_file_info(filepath, structure)
            if detailed_info:
                detailed_files.append(f"{filepath}\n{detailed_info}")
            else:
                detailed_files.append(filepath)

        detailed_files_str = "\n\n".join(detailed_files)
        formatted_structure = show_project_structure(structure).strip()

        refine_message = self.refine_files_prompt.format(
            problem_statement=problem_statement,
            structure=formatted_structure,
            initial_files=detailed_files_str,
        ).strip()

        # Log the refinement prompt
        self.logger.info(f"Refinement prompt:\n{refine_message}")

        # Use OpenAI's GPT-4o for refinement
        client = OpenAI()  # Initialize the OpenAI client

        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in code issue localization.",
                    },
                    {"role": "user", "content": refine_message},
                ],
            )
            refined_output = response.choices[0].message.content.strip()
            refined_files = self.extract_file_paths_from_result(refined_output)
            # Log the refined files
            self.logger.info(f"Refined files identified:\n{refined_files}")
            found_files = refined_files

        except Exception as e:
            # Log the error and return initial files
            self.logger.error(f"Failed to refine files due to an error: {str(e)}")
            return found_files  # Return initial files in case of failure

        return found_files

    def extract_file_paths_from_result(self, result):
        # Extract file paths from the LLM's response
        lines = result.splitlines()
        file_paths = []
        for line in lines:
            if (
                line.startswith("```") or not line.strip()
            ):  # Skip code block markers and empty lines
                continue
            path = line.strip("`")
            if path.endswith(".py"):  # Ensure it is a file path
                file_paths.append(path)
        return file_paths

    obtain_relevant_code_prompt = """
Please look through the following GitHub problem description and file paths and provide a set of locations that one would need to edit to fix the problem.

### GitHub Problem Description ###
{problem_statement}

###

### File: {file_name} ###
{file_content}

###

Please provide either the class, the function name or line numbers that need to be edited.
### Example 1:
```
class: MyClass
```
### Example 2:
```
function: my_function
```
### Example 3:
```
line: 10
line: 24
```

Return just the location(s)
"""
    file_content_template = """
### File: {file_name} ###
{file_content}
"""
    file_content_in_block_template = """
### File: {file_name} ###
```python
{file_content}
```
"""
    obtain_relevant_code_combine_top_n_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class names, function or method names, or exact line numbers that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class name, function or method name, or the exact line numbers that need to be edited.
### Examples:
```
full_path1/file1.py
line: 10
class: MyClass1
line: 51

full_path2/file2.py
function: MyClass2.my_method
line: 12

full_path3/file3.py
function: my_function
line: 24
line: 156
```

Return just the location(s)
"""
    obtain_relevant_code_combine_top_n_no_line_number_prompt = """
Please review the following GitHub problem description and relevant files, and provide a set of locations that need to be edited to fix the issue.
The locations can be specified as class, method, or function names that require modification.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide the class, method, or function names that need to be edited.
### Examples:
```
full_path1/file1.py
function: my_function1
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method
class: MyClass3

full_path3/file3.py
function: my_function2
```

Return just the location(s)
"""
    obtain_relevant_functions_from_compressed_files_prompt = """
Please look through the following GitHub problem description and the skeleton of relevant files.
Provide a thorough set of locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related functions and classes.

### GitHub Problem Description ###
{problem_statement}

###
{file_contents}

###

Please provide locations as either the class or the function name.
### Examples:
```
full_path1/file1.py
class: MyClass1

full_path2/file2.py
function: MyClass2.my_method

full_path3/file3.py
function: my_function
```

Return just the location(s)
"""
    obtain_relevant_functions_and_vars_from_compressed_files_prompt_more = """
Please look through the following GitHub Problem Description and the Skeleton of Relevant Files.
Identify all locations that need inspection or editing to fix the problem, including directly related areas as well as any potentially related global variables, functions, and classes.
For each location you provide, either give the name of the class, the name of a method in a class, the name of a function, or the name of a global variable.

### GitHub Problem Description ###
{problem_statement}

### Skeleton of Relevant Files ###
{file_contents}

###

Please provide the complete set of locations as either a class name, a function name, or a variable name.
Note that if you include a class, you do not need to list its specific methods.
You can include either the entire class or don't include the class name and instead include specific methods in the class.
### Examples:
```
full_path1/file1.py
function: my_function_1
class: MyClass1
function: MyClass2.my_method

full_path2/file2.py
variable: my_var
function: MyClass3.my_method

full_path3/file3.py
function: my_function_2
function: my_function_3
function: MyClass4.my_method_1
class: MyClass5
```

Return just the locations.
"""

    def __init__(
        self,
        instance_id,
        structure,
        problem_statement,
        model_name,
        backend,
        logger,
        match_partial_paths,
        **kwargs,
    ):
        super().__init__(instance_id, structure, problem_statement)
        self.max_tokens = 300
        self.model_name = model_name
        self.backend = backend
        self.logger = logger
        self.match_partial_paths = match_partial_paths

    def _parse_model_return_lines(self, content: str) -> list[str]:
        if content:
            return content.strip().split("\n")

    def localize(
        self, top_n=1, mock=False, match_partial_paths=False
    ) -> tuple[list, list, list, any]:
        # lazy import, not sure if this is actually better?
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        found_files = []

        message = self.obtain_relevant_files_prompt.format(
            problem_statement=self.problem_statement,
            structure=show_project_structure(self.structure).strip(),
        ).strip()
        self.logger.info(f"prompting with message:\n{message}")
        print(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)
        print("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]
        model_found_files = self._parse_model_return_lines(raw_output)

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        # sort based on order of appearance in model_found_files
        found_files = correct_file_paths(model_found_files, files, match_partial_paths)

        self.logger.info(raw_output)
        print(raw_output)

        return (
            found_files,
            {"raw_output_files": raw_output},
            traj,
        )

    def localize_function_for_files(
        self, file_names, mock=False
    ) -> tuple[list, dict, dict]:
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        files, classes, functions = get_full_file_paths_and_classes_and_functions(
            self.structure
        )

        max_num_files = len(file_names)
        while 1:
            # added small fix to prevent too many tokens
            contents = []
            for file_name in file_names[:max_num_files]:
                for file_content in files:
                    if file_content[0] == file_name:
                        content = "\n".join(file_content[1])
                        file_content = line_wrap_content(content)
                        contents.append(
                            self.file_content_template.format(
                                file_name=file_name, file_content=file_content
                            )
                        )
                        break
                else:
                    raise ValueError(f"File {file_name} does not exist.")

            file_contents = "".join(contents)
            if num_tokens_from_messages(file_contents, model) < MAX_CONTEXT_LENGTH:
                break
            else:
                max_num_files -= 1

        message = self.obtain_relevant_code_combine_top_n_prompt.format(
            problem_statement=self.problem_statement,
            file_contents=file_contents,
        ).strip()
        print(f"prompting with message:\n{message}")
        print("=" * 80)
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            loggger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_function_from_compressed_files(self, file_names, mock=False):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        # self.logger.info(f"Repository structure:\n{self.structure}")
        self.logger.info(f"Attempting to retrieve files:\n{file_names}")
        file_contents = get_repo_files(self.structure, file_names)
        compressed_file_contents = {
            fn: get_skeleton(code) for fn, code in file_contents.items()
        }
        contents = [
            self.file_content_in_block_template.format(file_name=fn, file_content=code)
            for fn, code in compressed_file_contents.items()
        ]
        file_contents = "".join(contents)
        template = (
            self.obtain_relevant_functions_and_vars_from_compressed_files_prompt_more
        )
        message = template.format(
            problem_statement=self.problem_statement, file_contents=file_contents
        )

        def message_too_long(message):
            return (
                num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH
            )

        while message_too_long(message) and len(contents) > 1:
            self.logger.info(f"reducing to \n{len(contents)} files")
            contents = contents[:-1]
            file_contents = "".join(contents)
            message = template.format(
                problem_statement=self.problem_statement, file_contents=file_contents
            )  # Recreate message

        if message_too_long(message):
            raise ValueError(
                "The remaining file content is too long to fit within the context length"
            )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(
                        message,
                        self.model_name,
                    ),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=0,
            batch_size=1,
        )
        traj = model.codegen(message, num_samples=1)[0]
        traj["prompt"] = message
        raw_output = traj["response"]

        model_found_locs = extract_code_blocks(raw_output)
        model_found_locs_separated = extract_locs_for_files(
            model_found_locs, file_names
        )

        self.logger.info(f"==== raw output ====")
        self.logger.info(raw_output)
        self.logger.info("=" * 80)
        self.logger.info(f"==== extracted locs ====")
        for loc in model_found_locs_separated:
            self.logger.info(loc)
        self.logger.info("=" * 80)

        print(raw_output)

        return model_found_locs_separated, {"raw_output_loc": raw_output}, traj

    def localize_line_from_coarse_function_locs(
        self,
        file_names,
        coarse_locs,
        context_window: int,
        add_space: bool,
        sticky_scroll: bool,
        no_line_number: bool,
        temperature: float = 0.0,
        num_samples: int = 1,
        mock=False,
    ):
        from agentless.util.api_requests import num_tokens_from_messages
        from agentless.util.model import make_model

        # Retrieve file contents for the given file names
        file_contents = {}
        for file_name in file_names:
            try:
                content = get_repo_files(self.structure, [file_name])
                file_contents.update(content)
            except AssertionError as e:
                self.logger.warning(f"Skipping missing file: {file_name} - {str(e)}")
                continue

        # Handle the case where file contents are not found or missing
        valid_files = []
        for pred_file in coarse_locs.keys():
            if pred_file not in file_contents:
                self.logger.warning(
                    f"File not found in retrieved contents: {pred_file}"
                )
            else:
                valid_files.append(pred_file)

        # Proceed only with valid files
        if not valid_files:
            self.logger.error("No valid files found for processing.")
            return [], {"raw_output_loc": ""}, {}

        # Construct the context for valid files
        topn_content, file_loc_intervals = construct_topn_file_context(
            coarse_locs,
            file_names,
            file_contents,
            self.structure,
            context_window=context_window,
            loc_interval=True,
            add_space=add_space,
            sticky_scroll=sticky_scroll,
            no_line_number=no_line_number,
        )
        # Choose the appropriate prompt template
        if no_line_number:
            template = self.obtain_relevant_code_combine_top_n_no_line_number_prompt
        else:
            template = self.obtain_relevant_code_combine_top_n_prompt

        # Format the message with the gathered content
        message = template.format(
            problem_statement=self.problem_statement, file_contents=topn_content
        )
        self.logger.info(f"prompting with message:\n{message}")
        self.logger.info("=" * 80)

        # Check if the message is within the model's context length
        if num_tokens_from_messages(message, self.model_name) >= MAX_CONTEXT_LENGTH:
            self.logger.error("Message exceeds the maximum context length.")
            return [], {"raw_output_loc": ""}, {}

        # Handle mock mode
        if mock:
            self.logger.info("Skipping querying model since mock=True")
            traj = {
                "prompt": message,
                "usage": {
                    "prompt_tokens": num_tokens_from_messages(message, self.model_name),
                },
            }
            return [], {"raw_output_loc": ""}, traj

        model = make_model(
            model=self.model_name,
            backend=self.backend,
            logger=self.logger,
            max_tokens=self.max_tokens,
            temperature=temperature,
            batch_size=num_samples,
        )
        raw_trajs = model.codegen(message, num_samples=num_samples)

        # Merge trajectories
        raw_outputs = [raw_traj["response"] for raw_traj in raw_trajs]
        traj = {
            "prompt": message,
            "response": raw_outputs,
            "usage": {  # merge token usage
                "completion_tokens": sum(
                    raw_traj["usage"]["completion_tokens"] for raw_traj in raw_trajs
                ),
                "prompt_tokens": sum(
                    raw_traj["usage"]["prompt_tokens"] for raw_traj in raw_trajs
                ),
            },
        }
        model_found_locs_separated_in_samples = []
        for raw_output in raw_outputs:
            model_found_locs = extract_code_blocks(raw_output)
            model_found_locs_separated = extract_locs_for_files(
                model_found_locs, file_names
            )
            model_found_locs_separated_in_samples.append(model_found_locs_separated)

            self.logger.info(f"==== raw output ====")
            self.logger.info(raw_output)
            self.logger.info("=" * 80)
            print(raw_output)
            print("=" * 80)
            self.logger.info(f"==== extracted locs ====")
            for loc in model_found_locs_separated:
                self.logger.info(loc)
            self.logger.info("=" * 80)
        self.logger.info("==== Input coarse_locs")
        coarse_info = ""
        for fn, found_locs in coarse_locs.items():
            coarse_info += f"### {fn}\n"
            if isinstance(found_locs, str):
                coarse_info += found_locs + "\n"
            else:
                coarse_info += "\n".join(found_locs) + "\n"
        self.logger.info("\n" + coarse_info)
        if len(model_found_locs_separated_in_samples) == 1:
            model_found_locs_separated_in_samples = (
                model_found_locs_separated_in_samples[0]
            )

        return (
            model_found_locs_separated_in_samples,
            {"raw_output_loc": raw_outputs},
            traj,
        )
