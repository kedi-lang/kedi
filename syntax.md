## Type Correctness for Field Annotations

Type correctness is secured thanks to venus.validator, it checks for type names and validates whether all type names are valid.
* Validation searches in typing, typing_extensions and builtins first to reach type.
* If it is a custom type, it will make a search in venus.types
* After validation, all the types names will convert into real python objects
  * Example: 'Literal[1, 2]' will transform to typing.Literal[1,2] as a python expression!
* Validation also determines marker schema (fields and annotations) to create pydantic models!

## User Inputs - Input Fields

User inputs can be defined with << input_field: annotation >> pattern.
* Annotation is not a must, defaulted to string or LLM determines based on context.
* Inputs can be retrieved from user inputs (cmd), environment variables or factory functions.
* You can use input utilities like this:
  * venus.inputs.user_input -> determines annotation, field name and creates a stdin process to retrieve value
    * Question('what is square of <<number: int>>', number=user_input)
  * venus.inputs.from_env -> retrieves field value from env variables, raises ValueError otherwise
    * Question('what is square of <<number: int>>', number=from_env)
  * venus.inputs.factory -> retrieves field value from factory function
    * Question('what is square of <<number: int>>', number=lambda: 27)

* Each input mechanism would validate type correctness for fields
* Inputs can also be specified via Marker definition
  * Question('what is square of <<number: int>>', number=27)


## Model Outputs - Output Fields

Model outputs can be defined in directly as textual in marker string with their type annotations.

* Annotation is a must for output fields to generate structured outputs, if you don't know the type,  use `contextual`, then LLM will find better type for the query and will replace it with new type.
* Output fields converted into pydantic models in runtime, because we are working on textual markers :)
* Output fields can be defaulted via `<field: annotation = OutputDefault>`


## LLM Invoking
There would be a proxy layer for pydantic_ai.Agent.run for invoking custom LLM calls by generatings runtime models with `create_model` based on marker schema.
* The generated model will be output_type of Agent.run call.
* The marker topic (non-marker side of texts, the template) will be passed as prompt and LLM will determine structured output for this query, thanks to PyAI.
